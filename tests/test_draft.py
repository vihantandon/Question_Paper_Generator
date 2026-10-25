"""
The language model client (pipeline/llm.py) and the draft step
(pipeline/nodes/draft.py).

    python3 -m unittest tests.test_draft -v      (from the repo root)

No API key, no network, no Docker: the OpenAI client is replaced by FakeChat,
which gives back replies written in the test (or raises the errors a real
service raises), and remembers every request it was sent.
"""
import copy
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieval  # noqa: E402
import settings  # noqa: E402
import tests.test_phase1 as P  # noqa: E402  (its fake Neo4j / Chroma / embedder)
from pipeline import llm as L  # noqa: E402
from pipeline.graph import build_graph  # noqa: E402
from pipeline.nodes import draft as D  # noqa: E402
from pipeline.resources import Resources  # noqa: E402


# ---------------------------------------------------------------- stand-ins
class FakeChat:
    """Stands in for OpenAI(...).chat: gives the queued replies in order. A
    dict is sent back as JSON text; an Exception is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []                  # a copy of every request's arguments
        self.completions = self

    def create(self, **kwargs):
        self.sent.append(copy.deepcopy(kwargs))
        if not self.replies:
            raise AssertionError("the model was asked more often than the test expected")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, SimpleNamespace):       # a whole reply object, as the SDK gives it
            return reply
        if isinstance(reply, dict):
            reply = json.dumps(reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


class ApiError(Exception):
    """Shaped like the OpenAI SDK's errors: a status_code and the response headers."""

    def __init__(self, status, message="", retry_after=None):
        super().__init__(message or f"Error code: {status}")
        self.status_code = status
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = SimpleNamespace(headers=headers)


class APITimeoutError(Exception):
    """Same class name as the SDK's time-out error (no status code)."""


def make_client(*replies, **options):
    chat, slept = FakeChat(*replies), []
    client = L.LLMClient(chat, "fake-model", sleep=slept.append, log=lambda *a: None, **options)
    return client, chat, slept


ASK = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}]


# ---------------------------------------------------------------- reading a reply
class TestParseJsonReply(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(L.parse_json_reply('{"a": 1}'), {"a": 1})

    def test_fences_and_chatter_around_the_json_are_ignored(self):
        self.assertEqual(L.parse_json_reply('Here it is:\n```json\n{"a": 1}\n```\nGood luck!'), {"a": 1})
        self.assertEqual(L.parse_json_reply('Sure! {"a": {"b": [1, 2]}} Hope this helps.'),
                         {"a": {"b": [1, 2]}})

    def test_reasoning_notes_are_dropped(self):
        reply = '<think>The user wants {a question}. Let me plan.</think>\n{"a": 1}'
        self.assertEqual(L.parse_json_reply(reply), {"a": 1})

    def test_a_reply_cut_off_while_thinking_has_no_answer(self):
        with self.assertRaisesRegex(ValueError, "only reasoning"):
            L.parse_json_reply('<think>still planning {"a": 1}')

    def test_no_object_no_answer(self):
        for reply, reason in ((None, "empty"), ("   ", "empty"), ("[1, 2]", "not a JSON object"),
                              ("The answer is 42.", "not a JSON object"), ('{"a": 1', "not a JSON object")):
            with self.assertRaisesRegex(ValueError, reason):
                L.parse_json_reply(reply)


# ---------------------------------------------------------------- the client
class TestLLMClient(unittest.TestCase):
    def test_returns_the_object_and_asks_for_json(self):
        client, chat, slept = make_client({"a": 1})
        self.assertEqual(client.complete_json(ASK, temperature=0.2), {"a": 1})
        self.assertEqual(chat.sent[0]["model"], "fake-model")
        self.assertEqual(chat.sent[0]["temperature"], 0.2)
        self.assertEqual(chat.sent[0]["response_format"], {"type": "json_object"})
        self.assertEqual((client.requests, slept), (1, []))

    def test_a_reply_that_is_not_json_goes_back_with_the_reason(self):
        client, chat, _ = make_client("I think a good question would be about trees.", {"a": 1})
        asked = list(ASK)
        self.assertEqual(client.complete_json(asked), {"a": 1})
        second = chat.sent[1]["messages"]
        self.assertEqual(second[:2], ASK)
        self.assertEqual(second[2], {"role": "assistant",
                                     "content": "I think a good question would be about trees."})
        self.assertIn("not a JSON object", second[3]["content"])
        self.assertEqual(asked, ASK)                          # the caller's list is not changed

    def test_only_the_last_rejected_reply_goes_back(self):
        # 21 Oct: a checker report rejected twice was sent back twice; the third
        # request was 8511 tokens, over Groq's free 8000-per-minute limit (413)
        client, chat, _ = make_client("no 1", "no 2", {"a": 1})
        client.complete_json(ASK)
        third = chat.sent[2]["messages"]
        self.assertEqual(len(third), len(ASK) + 2)
        self.assertEqual(third[len(ASK)], {"role": "assistant", "content": "no 2"})
        self.assertNotIn("no 1", str(third))

    def test_a_long_rejected_reply_is_not_sent_back(self):
        long_reply = '{"report": "' + "x" * (L.ECHO_LIMIT + 10) + '"}'
        client, chat, _ = make_client(long_reply, {"a": 1})

        def validate(obj):
            if "a" not in obj:
                raise ValueError('"outline" must have one entry for each of the 14 answer-outline lines')
            return obj
        client.complete_json(ASK, validate=validate)
        second = chat.sent[1]["messages"]
        self.assertEqual(len(second), len(ASK) + 1)
        self.assertIn("14 answer-outline lines", second[-1]["content"])
        self.assertIn("Reply again with the whole JSON object, corrected.", second[-1]["content"])

    def test_json_the_service_could_not_make_is_a_bad_reply_not_a_missing_json_mode(self):
        # 21 Oct, gpt-oss-20b on Groq: this 400 turned JSON mode off, then every reply came back empty
        failed = ApiError(400, "Error code: 400 - {'error': {'message': 'Failed to generate JSON. Please adjust "
                               "your prompt.', 'code': 'json_validate_failed'}}")
        client, chat, _ = make_client(failed, {"a": 1})
        self.assertEqual(client.complete_json(ASK), {"a": 1})
        self.assertTrue(client.json_mode)
        self.assertEqual(chat.sent[1]["response_format"], {"type": "json_object"})
        self.assertIn("could not make valid JSON", chat.sent[1]["messages"][-1]["content"])

    def test_a_model_without_json_mode_is_asked_in_the_prompt_only(self):
        client, chat, _ = make_client(ApiError(400, "response_format json_object is not supported by this model"),
                                      {"a": 1})
        self.assertEqual(client.complete_json(ASK), {"a": 1})
        self.assertFalse(client.json_mode)
        self.assertNotIn("response_format", chat.sent[1])

    def test_a_reply_cut_off_while_thinking_says_so_and_is_asked_again(self):
        cut = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")])
        client, chat, _ = make_client(cut, {"a": 1})
        logged = []
        client.log = logged.append
        self.assertEqual(client.complete_json(ASK), {"a": 1})
        self.assertIn("cut off: the model used up its output tokens while still thinking", logged[0])
        self.assertIn("LLM_REASONING_EFFORT=low", logged[0])
        self.assertEqual(len(chat.sent[1]["messages"]), len(ASK) + 1)     # nothing to send back but the reason

    def test_reasoning_effort_is_sent_and_dropped_when_the_model_does_not_take_it(self):
        client, chat, _ = make_client(ApiError(400, "`reasoning_effort` is not supported with this model"),
                                      {"a": 1}, reasoning_effort="low")
        self.assertEqual(client.complete_json(ASK), {"a": 1})
        self.assertEqual(chat.sent[0]["extra_body"], {"reasoning_effort": "low"})
        self.assertNotIn("extra_body", chat.sent[1])
        self.assertIsNone(client.reasoning_effort)
        client, chat, _ = make_client({"a": 1})
        client.complete_json(ASK)
        self.assertNotIn("extra_body", chat.sent[0])                       # not set: not sent

    def test_tokens_are_counted(self):
        used = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(total_tokens=1234))
        client, _, _ = make_client(used, used)
        client.complete_json(ASK)
        client.complete_json(ASK)
        self.assertEqual((client.requests, client.tokens), (2, 2468))
        other, _, _ = make_client()
        self.assertEqual(L.usage_line(client, None, client, other),
                         "2 request(s), 2,468 tokens to fake-model; 0 request(s), 0 tokens to fake-model")

    def test_a_request_too_large_is_explained_and_does_not_stop_the_run(self):
        client, _, _ = make_client(ApiError(413, "Request too large: tokens per minute (TPM): Limit 8000"))
        with self.assertRaises(L.LLMError) as stop:
            client.complete_json(ASK)
        self.assertNotIsInstance(stop.exception, L.LLMFatalError)
        self.assertIn("too large for fake-model's limit (413)", str(stop.exception))

    def test_validate_can_reject_a_reply_and_what_it_returns_is_used(self):
        def validate(obj):
            if obj["marks"] != 5:
                raise ValueError("the question must be worth exactly 5 marks")
            return {"checked": True}

        client, chat, _ = make_client({"marks": 4}, {"marks": 5})
        self.assertEqual(client.complete_json(ASK, validate=validate), {"checked": True})
        self.assertIn("exactly 5 marks", chat.sent[1]["messages"][-1]["content"])

    def test_gives_up_after_max_attempts(self):
        client, chat, _ = make_client("no", "still no", "never", max_attempts=3)
        with self.assertRaises(L.LLMError) as stop:
            client.complete_json(ASK)
        self.assertNotIsInstance(stop.exception, L.LLMFatalError)   # only this question failed
        self.assertIn("3 attempt", str(stop.exception))
        self.assertIn("not a JSON object", str(stop.exception))
        self.assertEqual(len(chat.sent), 3)

    def test_a_busy_service_is_waited_out(self):
        client, chat, slept = make_client(ApiError(429, "Too many requests"), ApiError(503), {"a": 1})
        self.assertEqual(client.complete_json(ASK), {"a": 1})
        self.assertEqual(slept, [2.0, 4.0])
        self.assertEqual(len(chat.sent), 3)

    def test_the_wait_the_service_asks_for_is_used(self):
        client, _, slept = make_client(ApiError(429, retry_after=7), {"a": 1})
        client.complete_json(ASK)
        self.assertEqual(slept, [7.0])

    def test_a_time_out_is_tried_again(self):
        client, _, slept = make_client(APITimeoutError("Request timed out."), {"a": 1})
        self.assertEqual(client.complete_json(ASK), {"a": 1})
        self.assertEqual(len(slept), 1)

    def test_no_connection_stops_the_run_after_two_more_tries(self):
        # every question would wait just as long, so the whole run stops
        client, chat, slept = make_client(*[APITimeoutError("Connection error.")] * 3)
        with self.assertRaises(L.LLMFatalError) as stop:
            client.complete_json(ASK)
        self.assertIn("internet", str(stop.exception))
        self.assertEqual((len(chat.sent), len(slept)), (3, 2))

    def test_still_busy_after_every_wait_fails_this_request_only(self):
        client, chat, slept = make_client(*[ApiError(503)] * 6)
        with self.assertRaises(L.LLMError) as stop:
            client.complete_json(ASK)
        self.assertNotIsInstance(stop.exception, L.LLMFatalError)
        self.assertEqual(len(slept), L.MAX_BUSY_RETRIES)

    def test_wrong_settings_stop_at_once(self):
        for status, words in ((401, "API key was rejected"), (402, "no credit"),
                              (403, "may not use"), (404, "model id or the base URL")):
            client, chat, slept = make_client(ApiError(status, "nope"))
            with self.assertRaises(L.LLMFatalError) as stop:
                client.complete_json(ASK)
            self.assertIn(words, str(stop.exception))
            self.assertIn("LLM_MODEL", str(stop.exception))
            self.assertEqual((len(chat.sent), slept), (1, []), status)

    def test_a_used_up_daily_limit_stops_at_once(self):
        for error in (ApiError(429, "Rate limit exceeded: free-models-per-day"),
                      ApiError(429, "slow down", retry_after=3600)):
            client, chat, slept = make_client(error)
            with self.assertRaises(L.LLMFatalError):
                client.complete_json(ASK)
            self.assertEqual((len(chat.sent), slept), (1, []))

    def test_a_model_without_json_mode_is_asked_without_it(self):
        client, chat, slept = make_client(
            ApiError(400, "response_format json_object is not supported by this model"), {"a": 1}, {"b": 2})
        self.assertEqual(client.complete_json(ASK), {"a": 1})
        self.assertEqual(client.complete_json(ASK), {"b": 2})
        self.assertIn("response_format", chat.sent[0])
        self.assertNotIn("response_format", chat.sent[1])
        self.assertNotIn("response_format", chat.sent[2])          # remembered
        self.assertEqual(slept, [])

    def test_any_other_bad_request_fails_this_request_only(self):
        client, _, _ = make_client(ApiError(400, "This model's maximum context length is 8192 tokens"))
        with self.assertRaises(L.LLMError) as stop:
            client.complete_json(ASK)
        self.assertNotIsInstance(stop.exception, L.LLMFatalError)


class TestMakeLLM(unittest.TestCase):
    def test_no_key_stops_with_what_to_do(self):
        with mock.patch.object(settings, "LLM_API_KEY", ""), self.assertRaises(SystemExit) as stop:
            L.make_llm()
        self.assertIn("OPENROUTER_API_KEY", str(stop.exception))
        self.assertIn("--no-llm", str(stop.exception))

    def test_the_client_uses_the_settings_and_does_its_own_waiting(self):
        made = {}
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = lambda **kwargs: made.update(kwargs) or SimpleNamespace(chat="CHAT")
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            client = L.make_llm(base_url="https://openrouter.ai/api/v1", model="some/model:free",
                                api_key="sk-test", timeout=30)
        self.assertEqual((client.chat, client.model), ("CHAT", "some/model:free"))
        self.assertEqual((made["api_key"], made["base_url"], made["timeout"], made["max_retries"]),
                         ("sk-test", "https://openrouter.ai/api/v1", 30, 0))
        self.assertIn("X-Title", made["default_headers"])


# ---------------------------------------------------------------- what the model is given
def source(id_, role, topic_id, text, kind="book", section="sec"):
    return {"id": id_, "role": role, "topic_id": topic_id, "text": text, "type": kind,
            "section": section, "distance": 0.1}


TREES = {"id": "DS_03", "name": "Trees", "semester": 3}
POINTERS = {"id": "SDF1_04", "name": "Pointers", "semester": 1, "hops": 2}
CHUNKS = [source("b1", "main", "DS_03", "A binary tree node has left and right children."),
          source("b2", "main", "DS_03", "Inorder traversal visits left, root, right."),
          source("t9__SDF1_04", "blend", "SDF1_04", "Q3. A pointer stores an address.", kind="tutorial")]


def state(**changes):
    s = {"current_topic": TREES, "blend_topic": POINTERS, "target_marks": 5, "target_difficulty": "hard",
         "topic_cos": [{"co_id": "DS:CO4", "short_id": "CO4", "description": "implement non-linear DS"}],
         "prompt_instruction": "Write a 5-mark question on: Trees. Blend in Pointers.",
         "relevant_chunks": CHUNKS, "grounding_warning": None,
         "style_examples": [{"marks": 5, "bloom_level": "Apply", "text": "**Q2.** Insert 7 into the BST.\n****"},
                            {"marks": None, "bloom_level": None, "text": "Q4. " + "x" * 5000}]}
    s.update(changes)
    return s


class TestPrompt(unittest.TestCase):
    def test_sources_are_numbered_and_say_what_they_are_about(self):
        system, user = D.build_messages(state())
        self.assertEqual((system["role"], user["role"]), ("system", "user"))
        # balanced (the default): built on the sources, standard course facts allowed
        self.assertIn("Build the question on the numbered SOURCES: its central idea must come from a "
                      "source about the main topic", system["content"])
        self.assertIn("SOURCES -- build the question on these", user["content"])
        strict_system, strict_user = D.build_messages(state(), grounding="strict")
        self.assertIn("Use ONLY facts from the numbered SOURCES", strict_system["content"])
        self.assertIn("SOURCES -- the only facts you may use", strict_user["content"])
        self.assertEqual(strict_system["content"], D.SYSTEM_PROMPT)
        text = user["content"]
        self.assertIn('[S1] main topic "Trees" -- book, section "sec"\n'
                      'A binary tree node has left and right children.', text)
        self.assertIn('[S2] main topic "Trees"', text)
        self.assertIn('[S3] prerequisite "Pointers" -- tutorial\nQ3. A pointer stores an address.', text)

    def test_task_outcomes_and_reply_format_are_given(self):
        text = D.build_messages(state())[1]["content"]
        self.assertTrue(text.startswith("TASK\nWrite a 5-mark question on: Trees."))
        self.assertIn("Total marks: 5.  Difficulty: hard.", text)
        self.assertIn("- CO4: implement non-linear DS", text)
        self.assertIn('PREREQUISITE TO BLEND IN: "Pointers" (semester 1)', text)
        self.assertIn('"marks": 5,', text)
        self.assertIn("where a correct answer needs Pointers", text)

    def test_past_questions_show_marks_and_long_ones_are_cut(self):
        text = D.build_messages(state())[1]["content"]
        self.assertIn("PAST EXAM QUESTIONS", text)
        self.assertIn("(5 marks, Apply)\nInsert 7 into the BST.\n", text)    # no "**Q2.**", no "****"
        self.assertIn("(? marks, Bloom level not given)", text)
        self.assertIn("[...]", text)
        self.assertNotIn("x" * (D.MAX_EXAMPLE_CHARS + 10), text)

    def test_no_blend_section_for_an_easy_question(self):
        text = D.build_messages(state(blend_topic=None, relevant_chunks=CHUNKS[:2]))[1]["content"]
        self.assertNotIn("PREREQUISITE", text)
        self.assertIn('"blend_used": ""', text)

    def test_feedback_on_a_rejected_draft_is_passed_on(self):
        plain = D.build_messages(state())[1]["content"]
        self.assertNotIn("REJECTED", plain)
        again = state(revision_feedback="the answer needs a fact that is in no source",
                      draft={"question": "Q. Old question text"})
        text = D.build_messages(again)[1]["content"]
        self.assertIn("YOUR PREVIOUS DRAFT WAS REJECTED\nReason: the answer needs a fact that is in no source", text)
        self.assertIn("Previous draft:\nQ. Old question text", text)


# ---------------------------------------------------------------- checking the reply
SOURCES = D.number_sources(CHUNKS)


def reply(**changes):
    r = {"question": "Given a BST stored with pointers, answer the following.",
         "parts": [{"label": "a", "text": "Draw the tree after inserting 4, 2, 6.", "marks": 2},
                   {"label": "(b)", "text": "Write the insert function using pointers.", "marks": 3}],
         "marks": 5, "answer_outline": ["root 4, left 2, right 6", "recursive insert with Node*"],
         "sources_used": ["S1", "S3"], "blend_used": "the insert function follows child pointers"}
    r.update(changes)
    return r


def check(obj, blend=POINTERS):
    return D.parse_draft(obj, marks=5, sources=SOURCES, blend=blend)


class TestTheAnswerKeyIsWorkedOut(unittest.TestCase):
    """21 Oct: an accepted rod-cutting answer key said "obtain R[8]=? (maximum revenue)"."""

    def test_a_result_left_open_is_sent_back(self):
        for step, open_ in (("compute R[1]..R[8] and obtain R[8]=? (maximum revenue)", "=?"),
                            ("R[8] = ???", "= ???"), ("the final cost is TBD", "TBD"),
                            ("the order is to be determined by the trace", "to be determined")):
            with self.assertRaises(ValueError) as e:
                check(reply(answer_outline=["root 4", step]))
            self.assertIn(f'answer_outline line 2 leaves its result open ("{open_}")', str(e.exception))

    def test_a_heading_line_joins_the_line_after_it(self):
        d = check(reply(answer_outline=["Sort the lengths: 1, 2, 3, 4", "DP table computation:",
                                        "j=1: R[1]=2", "j=2: R[2]=3", "The output is:", "13"]))
        self.assertEqual(d["answer_outline"], ["Sort the lengths: 1, 2, 3, 4", "DP table computation: j=1: R[1]=2",
                                               "j=2: R[2]=3", "The output is: 13"])

    def test_sources_cited_in_the_outline_but_forgotten_are_added(self):
        # 21 Oct: three requests were spent on "must also use a source about the prerequisite"
        d = check(reply(sources_used=["S1"], answer_outline=["sort the keys with counting sort (S3)",
                                                             "insert them (S2, S1); strings S9 and S1 stay"],
                        blend_used="sorting first (S3)"))
        self.assertEqual(d["sources_used"], ["S1", "S3", "S2"])
        d = check(reply(sources_used=[], answer_outline=["insert as in [S2]"]), blend=None)
        self.assertEqual(d["sources_used"], ["S2"])
        with self.assertRaisesRegex(ValueError, "names none of the given sources"):
            check(reply(sources_used=[], answer_outline=["strings S1 and S2 are compared"]), blend=None)

    def test_numbered_tasks_without_parts_are_sent_back(self):
        # 21 Oct: "Provide: 1. the sorted list 2. the recurrence ... 5. one optimal plan", no marks
        with self.assertRaisesRegex(ValueError, "numbered tasks .* write them as \"parts\""):
            check(reply(question="Rod of length 7.\nProvide:\n1. The sorted list.\n2. The recurrence.", parts=[]))
        code = "Find the output:\n1. int x = 5;\n2. printf(x);"                 # numbered code lines
        self.assertEqual(check(reply(question=code))["question"], code)
        self.assertIn("1. int x = 5;", check(reply(question=code, parts=[]))["question"])

    def test_question_marks_that_are_not_open_results_are_fine(self):
        for step in ("R[j] = max(p_i + R[j-i] - (i<j ? c : 0))", "x == y ? 1 : 0", "Is it stable? Yes."):
            check(reply(answer_outline=[step]))

    def test_the_writer_is_told_to_work_every_part_out(self):
        for prompt in (D.SYSTEM_PROMPT, D.BALANCED_SYSTEM_PROMPT):
            self.assertIn("6. The answer outline is the answer key: work every part out to its final answer",
                          prompt)
            self.assertIn("choose data where the method matters", prompt)
            self.assertIn("9. Reply with ONE JSON object", prompt)


class TestParseDraft(unittest.TestCase):
    def test_a_good_reply_comes_out_in_one_fixed_shape(self):
        d = check(reply(sources_used=["[S1]", "s3", "Source 2", "S9", "S1"]))
        self.assertEqual(d["parts"], [{"label": "a", "text": "Draw the tree after inserting 4, 2, 6.", "marks": 2},
                                      {"label": "b", "text": "Write the insert function using pointers.", "marks": 3}])
        self.assertEqual(d["sources_used"], ["S1", "S3", "S2"])     # unknown S9 and the repeat dropped
        self.assertEqual(d["source_ids"], ["b1", "t9__SDF1_04", "b2"])
        self.assertEqual((d["marks"], d["blend_used"]), (5, "the insert function follows child pointers"))

    def test_marks_written_as_text_or_decimals_are_read(self):
        self.assertEqual(check(reply(marks="5 marks"))["marks"], 5)
        self.assertEqual(check(reply(marks=5.0))["marks"], 5)
        self.assertEqual(check(reply(sources_used="S1, S3"))["sources_used"], ["S1", "S3"])

    def test_a_question_without_parts(self):
        d = check(reply(parts=[]))
        self.assertEqual(d["parts"], [])
        d = check(reply(parts=[{"text": "one", "marks": 2}, {"text": "two", "marks": 3}]))
        self.assertEqual([p["label"] for p in d["parts"]], ["a", "b"])

    def test_wrong_marks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 5 marks"):
            check(reply(marks=6))
        with self.assertRaisesRegex(ValueError, "exactly 5 marks"):
            check(reply(marks="five"))
        with self.assertRaisesRegex(ValueError, "add up to 4 marks, not 5"):
            check(reply(parts=[{"text": "one", "marks": 1}, {"text": "two", "marks": 3}]))
        with self.assertRaisesRegex(ValueError, "part 2"):
            check(reply(parts=[{"text": "one", "marks": 5}, {"text": "two"}]))

    def test_missing_question_or_outline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '"question" is empty'):
            check(reply(question="  "))
        with self.assertRaisesRegex(ValueError, "answer_outline"):
            check(reply(answer_outline=[]))
        self.assertEqual(check(reply(answer_outline="one line"))["answer_outline"], ["one line"])

    def test_it_must_cite_a_main_topic_source(self):
        with self.assertRaisesRegex(ValueError, "names none of the given sources"):
            check(reply(sources_used=["S7", "the textbook"]))
        with self.assertRaisesRegex(ValueError, r"main topic \(S1, S2\)"):
            check(reply(sources_used=["S3"]))

    def test_a_hard_question_must_really_use_the_prerequisite(self):
        with self.assertRaisesRegex(ValueError, r'prerequisite "Pointers" \(S3\)'):
            check(reply(sources_used=["S1", "S2"]))
        with self.assertRaisesRegex(ValueError, '"blend_used" is empty'):
            check(reply(blend_used=""))

    def test_no_prerequisite_needed_when_there_is_none_or_no_text_for_it(self):
        d = check(reply(sources_used=["S1"]), blend=None)
        self.assertEqual(d["blend_used"], "")
        only_main = D.number_sources(CHUNKS[:2])
        d = D.parse_draft(reply(sources_used=["S1"], blend_used=""), marks=5, sources=only_main, blend=POINTERS)
        self.assertEqual(d["sources_used"], ["S1"])


class TestDraftFormat(unittest.TestCase):
    """Format problems from the ALGO_06 drafts of 20 Oct."""

    def check(self, **changes):
        return check(reply(**changes), blend=None)

    def test_parts_written_out_again_in_the_question_are_sent_back(self):
        question = ("You are given arrays A and P.\n\n(a) Rearrange the prices by length using counting sort."
                    "\n\n(b) Design a DP algorithm for the maximum revenue.")
        with self.assertRaisesRegex(ValueError, r'"question" also writes out the parts \(a\), \(b\)'):
            self.check(question=question)

    def test_parts_written_out_again_word_for_word_are_cut_off_without_asking_again(self):
        # 21 Oct: a request was spent only on "question also writes out the parts"
        parts = [{"label": "a", "text": "Sort the pairs by length.", "marks": 2},
                 {"label": "b", "text": "Find the maximum revenue.", "marks": 3}]
        question = ("A rod of length 5 and these prices are given.\n\n(a) Sort the pairs by length. [2]\n"
                    "(b) Find the maximum revenue. (3 marks)")
        d = self.check(question=question, parts=parts)
        self.assertEqual(d["question"], "A rod of length 5 and these prices are given.")
        self.assertEqual([p["text"] for p in d["parts"]], ["Sort the pairs by length.", "Find the maximum revenue."])

    def test_labels_inside_a_sentence_are_not_parts(self):
        d = self.check(question="Analyse a binary heap created using (a) an array and (b) a complete tree.")
        self.assertEqual(len(d["parts"]), 2)

    def test_a_single_part_worth_all_the_marks_becomes_part_of_the_question(self):
        d = self.check(question="Sort the prices, then find the maximum revenue.",
                       parts=[{"label": "a", "text": "Give complete pseudocode.", "marks": 5}])
        self.assertEqual((d["question"], d["parts"]),
                         ("Sort the prices, then find the maximum revenue.\nGive complete pseudocode.", []))

    def test_backslash_n_written_as_text_becomes_a_line_break(self):
        d = self.check(question="The arrays are not ordered by length.\\n\\nSort them first.", parts=[])
        self.assertEqual(d["question"], "The arrays are not ordered by length.\n\nSort them first.")

    def test_backslash_n_in_c_code_is_left_alone(self):
        for question in ('Predict the output of printf("%d\\n", x); for x = 5.',     # inside quotes
                         'Trace the code:\nint x = 5;\nprintf("%d\\n", x);'):       # real line breaks too
            self.assertEqual(self.check(question=question, parts=[])["question"], question)


# ---------------------------------------------------------------- the node
class TestDraftNode(unittest.TestCase):
    def test_a_topic_without_grounding_text_is_not_sent_to_the_model(self):
        client, chat, _ = make_client()
        out = D.make_draft_node(client)(state(relevant_chunks=[], grounding_warning="No grounding text for DS_03"))
        self.assertEqual(out, {"draft": None, "draft_error": "not drafted: No grounding text for DS_03"})
        self.assertEqual(chat.sent, [])

    def test_a_good_reply_becomes_the_draft(self):
        client, chat, _ = make_client(reply())
        out = D.make_draft_node(client)(state())
        self.assertEqual(out["draft"]["source_ids"], ["b1", "t9__SDF1_04"])
        self.assertEqual(out["draft"]["model"], "fake-model")
        self.assertEqual((out["draft_error"], out["draft_attempts"]), (None, 1))
        self.assertEqual(chat.sent[0]["messages"], D.build_messages(state()))

    def test_a_reply_that_fails_a_check_is_sent_back_and_fixed(self):
        client, chat, _ = make_client(reply(marks=10), reply())
        out = D.make_draft_node(client)(state())
        self.assertEqual(out["draft"]["marks"], 5)
        self.assertIn("exactly 5 marks", chat.sent[1]["messages"][-1]["content"])

    def test_when_the_model_gives_up_the_error_is_kept_and_the_run_goes_on(self):
        client, _, _ = make_client("no", "no", "no")
        out = D.make_draft_node(client)(state(draft_attempts=1))
        self.assertIsNone(out["draft"])
        self.assertIn("no usable reply", out["draft_error"])
        self.assertEqual(out["draft_attempts"], 2)

    def test_wrong_settings_stop_the_run(self):
        client, _, _ = make_client(ApiError(401, "invalid key"))
        with self.assertRaises(L.LLMFatalError):
            D.make_draft_node(client)(state())


# ---------------------------------------------------------------- the whole graph
def run(topic_id, difficulty, marks, llm):
    """Draft only: the check and its loop are in tests/test_quality_check.py, classify in
    tests/test_classify.py."""
    res = Resources(driver=P.FakeDriver(P.GRAPH), grounding_collections=[P.BOOK],
                    pyq_collection=P.PYQS, embed_fn=P.fake_embed, llm=llm)
    return build_graph(res, check=False, classify=False).invoke({"current_topic_id": topic_id, "target_difficulty": difficulty,
                                                 "target_marks": marks})


class TestPipelineWithDraft(unittest.TestCase):
    def test_lookup_retrieve_and_draft_run_in_order(self):
        # the fake book: S1 = b1, S2 = b2 (Trees), S3 = b3 (Pointers, the blend)
        client, chat, _ = make_client(reply(sources_used=["S2", "S3"]))
        r = run("DS_03", "hard", 5, client)
        self.assertEqual([c["id"] for c in r["relevant_chunks"]], ["b1", "b2", "b3"])
        self.assertEqual(r["draft"]["source_ids"], ["b2", "b3"])
        self.assertEqual(r["draft_attempts"], 1)
        prompt = chat.sent[0]["messages"][1]["content"]
        self.assertIn("[S3] prerequisite \"Pointers\"", prompt)
        self.assertIn("Insert into a binary search tree", prompt)      # a past question from P.PYQS

    def test_without_a_model_the_graph_ends_after_retrieve(self):
        r = run("DS_03", "hard", 5, None)
        self.assertNotIn("draft", r)
        self.assertTrue(r["relevant_chunks"])

    def test_a_topic_without_text_costs_no_request(self):
        client, chat, _ = make_client()
        r = run("DS_07", "easy", 2, client)
        self.assertIsNone(r["draft"])
        self.assertIn("No grounding text for DS_07", r["draft_error"])
        self.assertEqual(chat.sent, [])


# ---------------------------------------------------------------- retrieval changes
class TestRetrievalForDrafting(unittest.TestCase):
    def test_grounding_chunks_carry_their_chroma_id(self):
        got = retrieval.fetch_grounding_chunks([P.BOOK], P.fake_embed, {"id": "DS_03", "name": "Trees"})
        self.assertEqual([c["id"] for c in got], ["b1", "b2"])

    def test_past_questions_of_about_the_same_marks_come_first(self):
        def q(id_, marks, vec):
            return (id_, f"question {id_}", {"topic_id": "DS_03", "subject": "DS", "marks": marks}, vec)

        pyqs = P.FakeCollection("pyq_bank", [q("closest_2m", 2, [1.0, 0.0]), q("next_6m", 6, [0.9, 0.1]),
                                             q("far_5m", 5, [0.5, 0.5]), q("other_10m", 10, [0.0, 1.0])])

        def ids(**kw):
            return [h["id"] for h in retrieval.fetch_similar_pyqs(pyqs, P.fake_embed, "Trees", "DS_03", **kw)]

        self.assertEqual(ids(n=3), ["closest_2m", "next_6m", "far_5m"])              # no target: similarity
        self.assertEqual(ids(n=3, marks=5), ["next_6m", "far_5m", "closest_2m"])     # 6 and 5 are within 1
        self.assertEqual(ids(n=2, marks=10), ["other_10m", "closest_2m"])
        self.assertEqual(ids(n=2, marks=5, exclude_ids=["next_6m"]), ["far_5m", "closest_2m"])


# The DS_03 case, in three dimensions: x = "the topic's name" (advanced
# handbook text is closest to it), y = "building a heap", z = "heap in an array".
VECTORS = {"Trees": [1.0, 0.0, 0.0], "Pointers (used within Trees)": [0.0, 0.0, 1.0]}
ADVANCED_1 = ("adv1", "Meldable priority queues: amortized analysis.", [1.0, 0.0, 0.0])
ADVANCED_2 = ("adv2", "Symmetric min-max heaps for double-ended queues.", [0.95, 0.05, 0.0])
HEAP_TEXT = ("heap", "A min heap is a complete binary tree in which each key is <= its children.", [0.0, 1.0, 0.0])
ARRAY_TEXT = ("array", "Stored in an array, node i has children 2i and 2i+1, parent i/2.", [0.0, 0.0, 1.0])
BUILD_Q = ("t_build", "Q2. Construct a Min Binary Heap using 10, 22, 30, 40.", [0.4, 0.6, 0.0])
ARRAY_Q = ("t_array", "Q3. Analyse the Binary Heap created using (a) array (b) tree.", [0.3, 0.0, 0.7])


def embed(text):
    """Queries are the topic name or a tutorial question's own text."""
    rows = {t: v for _, t, v in (ADVANCED_1, ADVANCED_2, HEAP_TEXT, ARRAY_TEXT, BUILD_Q, ARRAY_Q)}
    return VECTORS.get(text) or rows[text]


def collection(name, topic, *rows):
    """rows: (id, text, vector); every row tagged `topic`, usable."""
    return P.FakeCollection(name, [P.chunk(i, topic, t, v) for i, t, v in rows])


HANDBOOK = collection("book_content", "DS_03", ADVANCED_1, ADVANCED_2, HEAP_TEXT, ARRAY_TEXT)
TUTORIALS = collection("tut_content", "DS_03", BUILD_Q, ARRAY_Q)


def picked(collections, **kw):
    got = retrieval.fetch_grounding_chunks(collections, embed, TREES, **kw)
    return [(c["id"], c["found_by"]) for c in got]


class TestTutorialQuestionsBringTheirExplanations(unittest.TestCase):
    def test_each_tutorial_question_brings_the_book_text_that_explains_it(self):
        # by topic name alone the model got amortized analysis and min-max heaps,
        # and wrote the heap's index formulas from its own memory
        self.assertEqual(picked([HANDBOOK, TUTORIALS]),
                         [("adv1", "topic"),
                          ("heap", "explains t_build"), ("t_build", "tutorial"),
                          ("array", "explains t_array"), ("t_array", "tutorial")])

    def test_searching_by_the_topic_name_alone_misses_both_explanations(self):
        ids = [i for i, _ in picked([HANDBOOK, TUTORIALS], n_main=4, min_tutorial=0)]
        self.assertEqual(ids, ["adv1", "adv2", "t_build", "t_array"])

    def test_one_tutorial_question_brings_one_explanation_and_the_rest_is_by_topic(self):
        one = collection("tut_content", "DS_03", BUILD_Q)
        self.assertEqual(picked([HANDBOOK, one], n_main=4),
                         [("adv1", "topic"), ("adv2", "topic"),
                          ("heap", "explains t_build"), ("t_build", "tutorial")])

    def test_without_tutorial_questions_it_is_all_by_topic(self):
        self.assertEqual([i for i, _ in picked([HANDBOOK], n_main=3)], ["adv1", "adv2", "heap"])

    def test_without_book_text_the_tutorial_questions_still_come(self):
        # e.g. ALGO_05 Backtracking: tutorial questions only
        self.assertEqual(picked([TUTORIALS]), [("t_build", "tutorial"), ("t_array", "tutorial")])

    def test_two_questions_never_bring_the_same_text(self):
        build_again = ("t_build2", "Q4. Build a max heap from 5, 8, 2.", [0.45, 0.55, 0.0])
        tut = collection("tut_content", "DS_03", BUILD_Q, build_again)
        with mock.patch.dict(VECTORS, {build_again[1]: build_again[2]}):
            got = picked([HANDBOOK, tut], n_main=4)
        self.assertEqual(got, [("heap", "explains t_build2"), ("t_build2", "tutorial"),
                               ("adv2", "explains t_build"), ("t_build", "tutorial")])

    # the prerequisite is searched with its name AND the main topic's first source
    BLEND_QUERY = "Pointers (used within Trees): Meldable priority queues: amortized analysis."

    def test_the_prerequisite_text_closest_to_the_main_sources_is_used(self):
        # 25 Oct: by its name alone, Linear DS gave heaps an STL linked-list exercise and an
        # XOR linked list; the writer could not use them and DS_03 medium failed three times
        book = collection("book_content", "SDF1_04", ("p_far", "Pointer arithmetic.", [1.0, 0.0, 0.0]),
                          ("p_swap", "Passing addresses lets a function swap two ints.", [0.0, 0.1, 0.9]))
        swap_q = ("q_swap", "Q1. Swap two ints with pointers.", [0.0, 0.2, 0.8])
        tut = collection("tut_content", "SDF1_04", swap_q)
        # by the name alone the search would land on "Pointer arithmetic." (p_far)
        with mock.patch.dict(VECTORS, {self.BLEND_QUERY: [0.0, 0.1, 0.9], "Pointers (used within Trees)": [1.0, 0.0, 0.0]}):
            got = retrieval.fetch_grounding_chunks([HANDBOOK, TUTORIALS, book, tut], embed, TREES,
                                                   blend_topic={"id": "SDF1_04", "name": "Pointers"})
        self.assertEqual([(c["id"], c["found_by"]) for c in got if c["role"] == "blend"],
                         [("p_swap", "topic"), ("q_swap", "topic")])

    def test_a_tutorial_question_and_its_explanation_can_still_be_asked_for(self):
        book = collection("book_content", "SDF1_04", ("p_far", "Pointer arithmetic.", [1.0, 0.0, 0.0]),
                          ("p_swap", "Passing addresses lets a function swap two ints.", [0.0, 0.1, 0.9]))
        swap_q = ("q_swap", "Q1. Swap two ints with pointers.", [0.0, 0.2, 0.8])
        tut = collection("tut_content", "SDF1_04", swap_q)
        with mock.patch.dict(VECTORS, {swap_q[1]: swap_q[2], self.BLEND_QUERY: [0.0, 0.0, 1.0]}):
            got = retrieval.fetch_grounding_chunks([HANDBOOK, TUTORIALS, book, tut], embed, TREES,
                                                   blend_topic={"id": "SDF1_04", "name": "Pointers"},
                                                   min_tutorial_blend=1)
        self.assertEqual([(c["id"], c["found_by"]) for c in got if c["role"] == "blend"],
                         [("p_swap", "explains q_swap"), ("q_swap", "tutorial")])


def graded(id_, text, vec, advanced):
    """A book row with the "advanced" flag tag_topics.py sets."""
    meta = {"topic_id": "DS_03", "needs_review": False, "subject": "DS", "type": "book",
            "section_title": "sec", "advanced": advanced}
    return (id_, text, meta, vec)


# the interval-heap section is closest to the question, the core heap text a little farther
GRADED_BOOK = P.FakeCollection("book_content", [
    graded("interval", "Interval heaps: each node holds a pair.", [0.45, 0.55, 0.0], True),
    graded("core_heap", "A min heap is a complete binary tree ...", [0.0, 1.0, 0.0], False),
    graded("adv_topic", "Meldable priority queues.", [1.0, 0.0, 0.0], True),
    graded("core_topic", "Priority queues are implemented with heaps.", [0.9, 0.1, 0.0], False)])
BUILD_ONLY = collection("tut_content", "DS_03", BUILD_Q)


def graded_ids(**kw):
    got = retrieval.fetch_grounding_chunks([GRADED_BOOK, BUILD_ONLY], embed, TREES, n_main=3, **kw)
    return [c["id"] for c in got]


class TestCoreTextForEasyAndMediumQuestions(unittest.TestCase):
    def test_core_only_leaves_out_advanced_chapters(self):
        self.assertEqual(graded_ids(core_only=True), ["core_topic", "core_heap", "t_build"])

    def test_otherwise_advanced_text_can_be_picked(self):
        self.assertEqual(graded_ids(), ["adv_topic", "interval", "t_build"])

    def test_a_topic_with_only_advanced_book_text_still_gets_it(self):
        only_advanced = P.FakeCollection("book_content", [r for r in GRADED_BOOK.rows if r[2]["advanced"]])
        got = retrieval.fetch_grounding_chunks([only_advanced, BUILD_ONLY], embed, TREES, n_main=3,
                                               core_only=True)
        self.assertEqual([c["id"] for c in got], ["adv_topic", "interval", "t_build"])

    def test_vectors_tagged_before_the_flag_existed_are_used_as_before(self):
        # P.BOOK rows have no "advanced" key: nothing matches core_only, so all text is used
        got = retrieval.fetch_grounding_chunks([P.BOOK], P.fake_embed, {"id": "DS_03", "name": "Trees"},
                                               core_only=True)
        self.assertEqual([c["id"] for c in got], ["b1", "b2"])

    def test_the_retrieve_step_uses_core_text_for_easy_and_medium_only(self):
        from pipeline.nodes.retrieve import make_retrieve_node
        node = make_retrieve_node([GRADED_BOOK, BUILD_ONLY], None, embed, n_main=3)
        state = {"current_topic": TREES, "blend_topic": None, "target_marks": 5}
        for difficulty, want in (("easy", "core_heap"), ("medium", "core_heap"), ("hard", "interval")):
            chunks = node(dict(state, target_difficulty=difficulty))["relevant_chunks"]
            self.assertIn(want, [c["id"] for c in chunks], difficulty)


class TestCleanExample(unittest.TestCase):
    def test_question_number_and_left_over_asterisks_go(self):
        for raw in ("**Q3.** A binomial heap H3 is built.\n****", "Q.3 A binomial heap H3 is built.",
                    "Q3) A binomial heap H3 is built.", "  Question 3: A binomial heap H3 is built."):
            self.assertEqual(D.clean_example(raw), "A binomial heap H3 is built.", raw)

    def test_code_and_parts_are_kept(self):
        raw = "**Q2.** Find the output.\n****\n\n```cpp\nint **p = &q;\n```\na) Trace it.\nb) Fix it."
        self.assertEqual(D.clean_example(raw),
                         "Find the output.\n\n```cpp\nint **p = &q;\n```\na) Trace it.\nb) Fix it.")


if __name__ == "__main__":
    unittest.main()

"""
The quality check (pipeline/nodes/quality_check.py) and the draft <-> check
loop (pipeline/graph.py).

    python3 -m unittest tests.test_quality_check -v      (from the repo root)

No API key, no network, no Docker: the language model is FakeChat from
tests/test_draft.py, which gives back replies written here.
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_draft as T  # noqa: E402  (FakeChat, make_client, ApiError, sources, drafts)
import tests.test_phase1 as P  # noqa: E402  (fake Neo4j / Chroma / embedder)
from pipeline import llm as L  # noqa: E402
from pipeline.graph import accepted, build_graph, print_checks  # noqa: E402
from pipeline.nodes import quality_check as Q  # noqa: E402
from pipeline.nodes.draft import number_sources  # noqa: E402
from pipeline.resources import Resources  # noqa: E402


def check_reply(lines, question=(), **flags):
    """What a checker sends back. lines: for each answer-outline line, its
    claims as (claim, source, quote); question: the question text's claims."""
    def claim(c):
        return c if isinstance(c, dict) else {"claim": c[0], "source": c[1], "quote": c[2]}
    r = {"question_claims": [claim(c) for c in question],
         "outline": [{"line": n, "claims": [claim(c) for c in line]} for n, line in enumerate(lines, 1)],
         "answerable": True, "answerable_reason": "", "single_reading": True, "ambiguity": "",
         "gives_away": False, "gives_away_reason": "",
         "marks_fit": True, "marks_reason": "", "blend_needed": True, "blend_reason": "",
         "feedback": ""}
    r.update(flags)
    return r


def W(claim, builds_on, follows=True):
    """A WORKING claim: what the student works out, built on these sources."""
    return {"claim": claim, "kind": "working", "builds_on": builds_on, "follows": follows}


TWO_CHILDREN = ("A node has two children", "S1", "has left and right children")
POINTER = ("A pointer stores an address", "S3", "A pointer stores an address")


# ---------------------------------------------------------------- is the quote really there?
CLRS = ("Problem 6-2 asks: e. Give an efficient implementation of INCREASE-KEY(A, i, k). An-\n"
        "alyze its running time in terms of d and n. The MAX-HEAP-INSERT, HEAP-EXTRACT-MAX,\n"
        "HEAP-INCREASE-KEY, and HEAP-MAXIMUM procedures, which run in O(lg n) time, allow the heap")


class TestQuoteInSource(unittest.TestCase):
    def test_exact_words(self):
        self.assertTrue(Q.quote_in_source("procedures, which run in O(lg n) time", CLRS))

    def test_case_spacing_punctuation_and_broken_words_do_not_matter(self):
        self.assertTrue(Q.quote_in_source("PROCEDURES which run in   O lg n time", CLRS))
        self.assertTrue(Q.quote_in_source("Analyze its running time in terms of d and n", CLRS))

    def test_left_out_words_may_be_marked_with_dots(self):
        self.assertTrue(Q.quote_in_source("The MAX-HEAP-INSERT ... procedures, which run in O(lg n) time", CLRS))
        self.assertTrue(Q.quote_in_source("The MAX-HEAP-INSERT … allow the heap", CLRS))
        self.assertFalse(Q.quote_in_source("allow the heap ... The MAX-HEAP-INSERT", CLRS))   # wrong order

    def test_words_that_are_not_there(self):
        # the model's own extract-max steps from the draft that prompted this check
        self.assertFalse(Q.quote_in_source("replace A[0] with the last element and max-heapify", CLRS))
        self.assertFalse(Q.quote_in_source("runs in O(lg n)", "it runs in O(n lg n) time"))   # n lg n != lg n

    def test_pseudocode_line_numbers_in_between_do_not_matter(self):
        # ALGO_06, 20 Oct: "if n == 0 return 0" was quoted correctly from CLRS's CUT-ROD,
        # but the chunk has the line number "2" on a line of its own in between
        cut_rod = "CUT-ROD(p, n)\n1\nif n == 0\n2\nreturn 0\n3\nq = −∞\n4\nfor i = 1 to n"
        self.assertTrue(Q.quote_in_source("if n == 0 return 0", cut_rod))
        self.assertTrue(Q.quote_in_source("1 if n == 0 2 return 0", cut_rod))       # numbers copied too
        self.assertTrue(Q.quote_in_source("if n == 0 ... for i = 1 to n", cut_rod))
        self.assertFalse(Q.quote_in_source("if n == 1 return 0", cut_rod))
        self.assertFalse(Q.quote_in_source("return 0 if n == 0", cut_rod))         # wrong order

    def test_too_short_or_empty_proves_nothing(self):
        for quote in ("", "   ", "heap", "...", None):
            self.assertFalse(Q.quote_in_source(quote, CLRS), quote)


# ---------------------------------------------------------------- the checker's report
SOURCES = number_sources(T.CHUNKS)      # S1, S2 about Trees; S3 about Pointers (the blend)


class TestParseReport(unittest.TestCase):
    def parse(self, reply, blend=T.POINTERS, outline_lines=2):
        return Q.parse_report(reply, sources=SOURCES, blend=blend, outline_lines=outline_lines)

    def test_a_claim_counts_only_when_its_quote_is_in_the_source_it_names(self):
        r = self.parse(check_reply(
            [[("Inorder is left, root, right", "[S2]", "visits left, root, right"),
              ("A pointer stores an address", "S1", "A pointer stores an address")],    # S3 says it, not S1
             [("Heaps use arrays", None, ""),
              ("Something from nowhere", "S9", "anything at all here")]],
            question=[TWO_CHILDREN]))
        self.assertEqual([(c["line"], c["source"], c["supported"]) for c in r["claims"]],
                         [(None, "S1", True), (1, "S2", True), (1, "S1", False), (2, None, False),
                          (2, None, False)])

    def test_every_outline_line_must_have_its_claims_listed(self):
        # the checker used to pick which facts to list, and skipped "better cache locality"
        with self.assertRaisesRegex(ValueError, "one entry for each of the 2 answer-outline lines"):
            self.parse(check_reply([[TWO_CHILDREN]]))
        with self.assertRaisesRegex(ValueError, "outline line 2 has no claims"):
            self.parse(check_reply([[TWO_CHILDREN], []]))
        self.assertEqual(len(self.parse(check_reply([[TWO_CHILDREN]]), outline_lines=1)["claims"]), 1)

    def test_flags_written_as_text_are_read(self):
        r = self.parse(check_reply([[TWO_CHILDREN], [POINTER]], answerable="true", single_reading="False",
                                   gives_away="false"))
        self.assertEqual((r["answerable"], r["single_reading"], r["gives_away"]), (True, False, False))

    def test_an_unusable_report_is_sent_back(self):
        good = check_reply([[TWO_CHILDREN], [POINTER]])
        bad = [dict(good, answerable=None),                                    # no verdict
               dict(good, gives_away=None),
               dict(good, outline="all fine"),
               dict(good, question_claims="none"),
               check_reply([[("", "S1", "children")], [POINTER]])]              # empty claim
        for reply in bad:
            with self.assertRaises(ValueError):
                self.parse(reply)
        with self.assertRaisesRegex(ValueError, "blend_needed"):
            self.parse(dict(good, blend_needed=None))
        self.assertIsNone(self.parse(dict(good, blend_needed=None), blend=None)["blend_needed"])


class TestProblems(unittest.TestCase):
    def report(self, **flags):
        lines = flags.pop("lines", [[TWO_CHILDREN], [POINTER]])
        return Q.parse_report(check_reply(lines, **flags), sources=SOURCES, blend=T.POINTERS,
                              outline_lines=len(lines))

    def test_a_good_draft_has_no_problems(self):
        self.assertEqual(Q.problems(self.report(), T.POINTERS), [])

    def test_each_kind_of_problem_is_named_for_the_writer(self):
        found = Q.problems(self.report(
            lines=[[("Heaps are stored in arrays", None, "")],
                   [("Extract-max moves the last element to the root", "S2", "move the last element to the root")]],
            answerable=False, answerable_reason="part (b) needs the heapify procedure",
            single_reading=False, ambiguity="part (b) 'how could it be adapted' has many answers",
            gives_away=True, gives_away_reason="the stem says deletion is O(n); (a) asks why",
            marks_fit=False, marks_reason="a full program for 2 marks",
            blend_needed=False, blend_reason="pointers are never used"), T.POINTERS)
        text = " ".join(found)
        self.assertEqual(len(found), 7)
        for words in ('"Heaps are stored in arrays" (outline line 1: in no source)',
                      '"Extract-max moves the last element to the root" (outline line 2: S2 does not say this',
                      "part (b) needs the heapify procedure", "'how could it be adapted' has many answers",
                      "already gives away what a part asks for: the stem says deletion is O(n)",
                      "a full program for 2 marks", 'does not really need "Pointers": pointers are never used',
                      "not built on the course material: none of its facts is quoted from a source about "
                      "the main topic (S1, S2)"):
            self.assertIn(words, text)

    def test_the_cache_locality_line_that_slipped_through_is_caught(self):
        # the accepted DS_03 draft of 20 Oct: outline line 2 gave advantages no source states
        found = Q.problems(self.report(lines=[
            [TWO_CHILDREN],
            [POINTER, ("An array avoids the memory overhead of explicit pointers", None, ""),
             ("Array storage gives better cache locality", None, "")]]), T.POINTERS)
        self.assertEqual(len(found), 1)
        self.assertIn('"Array storage gives better cache locality" (outline line 2: in no source)', found[0])


# ---------------------------------------------------------------- the node
def draft_state(**changes):
    s = T.state(draft={"question": "Given a BST stored with pointers, answer the following.",
                       "parts": [{"label": "a", "text": "Draw it.", "marks": 2},
                                 {"label": "b", "text": "Write insert.", "marks": 3}],
                       "marks": 5, "answer_outline": ["root 4", "recursive insert"],
                       "sources_used": ["S1", "S3"], "source_ids": ["b1", "t9__SDF1_04"],
                       "blend_used": "insert follows pointers"})
    s.update(changes)
    return s


GOOD = check_reply([[TWO_CHILDREN], [POINTER]])
BAD = check_reply([[TWO_CHILDREN], [("Insert is recursive", "S2", "insert works recursively")]],
                  feedback="Ask for the traversal order S2 gives instead.")


class TestQualityCheckNode(unittest.TestCase):
    def test_a_good_draft_passes(self):
        client, chat, _ = T.make_client(GOOD)
        out = Q.make_quality_check_node(client)(draft_state())
        self.assertTrue(out["quality_passed"])
        self.assertIsNone(out["revision_feedback"])
        self.assertEqual(out["quality"]["problems"], [])
        self.assertEqual(len(out["quality_reports"]), 1)
        self.assertEqual(chat.sent[0]["temperature"], 0.0)
        prompt = chat.sent[0]["messages"][1]["content"]
        for words in ("QUESTION TO CHECK (5 marks)", "(a) Draw it.   [2 marks]",
                      "ANSWER OUTLINE (2 lines", "1. root 4\n2. recursive insert",
                      "ONE entry for EACH of the 2 outline lines", "5. gives_away:", "how could X be adapted",
                      'PREREQUISITE THE QUESTION MUST NEED: "Pointers"', "[S3]\nQ3. A pointer stores an address."):
            self.assertIn(words, prompt)

    def test_a_bad_draft_fails_with_feedback_for_the_writer(self):
        # strict: a fact in the answer outline must be quoted from a source
        client, _, _ = T.make_client(BAD)
        out = Q.make_quality_check_node(client, grounding="strict")(draft_state(quality_reports=[{"older": 1}]))
        self.assertFalse(out["quality_passed"])
        self.assertIn('"Insert is recursive" (outline line 2: S2 does not say this', out["revision_feedback"])
        self.assertIn("Checker's advice: Ask for the traversal order S2 gives instead.", out["revision_feedback"])
        self.assertEqual(len(out["quality_reports"]), 2)

    def test_no_draft_means_nothing_to_check_and_no_request(self):
        client, chat, _ = T.make_client()
        self.assertEqual(Q.make_quality_check_node(client)(T.state(draft=None)),
                         {"quality": None, "quality_passed": False})
        self.assertEqual(chat.sent, [])

    def test_the_prerequisite_is_only_demanded_when_there_was_text_about_it(self):
        client, chat, _ = T.make_client(dict(GOOD, blend_needed=None))
        out = Q.make_quality_check_node(client)(draft_state(relevant_chunks=T.CHUNKS[:2]))
        self.assertNotIn("PREREQUISITE", chat.sent[0]["messages"][1]["content"])
        self.assertIsNone(out["quality"]["blend_needed"])

    def test_when_the_checker_gives_up_the_draft_is_not_passed_and_not_rewritten(self):
        client, _, _ = T.make_client("no", "no", "no")
        out = Q.make_quality_check_node(client)(draft_state())
        self.assertFalse(out["quality_passed"])
        self.assertIsNone(out["revision_feedback"])
        self.assertIn("error", out["quality"])

    def test_wrong_settings_stop_the_run(self):
        client, _, _ = T.make_client(T.ApiError(401, "bad key"))
        with self.assertRaises(L.LLMFatalError):
            Q.make_quality_check_node(client)(draft_state())


# ---------------------------------------------------------------- the draft that started this
HANDBOOK_3_7 = ("A priority queue can be implemented by a simple, unordered linked list. Insertions can be\n"
                "performed in O(1) time. However, a deletion requires a search for the element with the\n"
                "largest priority followed by its removal. The search requires time linear in the length of\n"
                "the linked list. When a max heap is used, both of these operations can be performed in\n"
                "O(log n) time.")


class TestTheExtractMaxDraft(unittest.TestCase):
    """The DS_03 medium draft of 20 Oct: part (a) came from the sources, part
    (b)'s extract-max steps and underflow check from the model's memory. A
    checker that claims S2 supports them is not believed: the words are not there."""

    def test_part_b_is_caught_and_part_a_passes(self):
        sources = number_sources([T.source("DS_051_00", "main", "DS_03", HANDBOOK_3_7),
                                  T.source("APS_009_06", "main", "DS_03", CLRS)])
        report = Q.parse_report(check_reply([
            [("An unordered linked list needs a linear search to delete the largest", "S1",
              "a deletion requires a search for the element with the largest priority"),
             ("With a max heap, deletion takes O(log n)", "S1",
              "When a max heap is used, both of these operations can be performed in O(log n) time")],
            [("Extract-Max moves A[n-1] to A[0] and max-heapifies", "S2",
              "HEAP-EXTRACT-MAX replaces the root with the last element and calls MAX-HEAPIFY")],
            [("Extract-Max on an empty heap is an underflow error", None, "")]]),
            sources=sources, blend=None, outline_lines=3)
        self.assertEqual([c["supported"] for c in report["claims"]], [True, True, False, False])
        feedback = " ".join(Q.problems(report))
        self.assertIn('"Extract-Max moves A[n-1] to A[0] and max-heapifies" (outline line 2: S2 does not', feedback)
        self.assertIn('"Extract-Max on an empty heap is an underflow error" (outline line 3: in no source)', feedback)


# ---------------------------------------------------------------- knowledge and working
class TestKnowledgeAndWorking(unittest.TestCase):
    def report(self, lines, **flags):
        return Q.parse_report(check_reply(lines, **flags), sources=SOURCES, blend=None,
                              outline_lines=len(lines))

    def test_working_needs_no_quote_when_it_follows_from_real_sources(self):
        r = self.report([[TWO_CHILDREN], [W("Insert 4, then 2 to its left, then 6 to its right", ["S1"])]])
        self.assertEqual([(c["kind"], c["supported"]) for c in r["claims"]],
                         [("knowledge", True), ("working", True)])
        self.assertEqual(Q.problems(r), [])

    def test_a_working_step_that_does_not_follow_fails(self):
        r = self.report([[TWO_CHILDREN], [W("Put 2 to the right of 4", ["S1"], follows=False)]])
        found = Q.problems(r)
        self.assertEqual(len(found), 1)
        self.assertIn('These steps are wrong, or do not follow from the sources: "Put 2 to the right of 4" '
                      '(outline line 2: does not follow from S1)', found[0])

    def test_working_must_build_on_a_real_source(self):
        for builds_on in ([], ["S9"], None):
            r = self.report([[W("Some step", builds_on)]])
            self.assertIn('"Some step" (outline line 1: builds on no source)', Q.problems(r)[0])

    def test_how_the_checker_writes_it_does_not_matter(self):
        r = self.report([[{"claim": "Step", "kind": "Working", "builds_on": "S1, [S3]", "follows": "true"}]])
        c = r["claims"][0]
        self.assertEqual((c["kind"], c["builds_on"], c["supported"]), ("working", ["S1", "S3"], True))

    def test_without_a_kind_a_claim_is_knowledge_and_needs_its_quote(self):
        r = self.report([[{"claim": "Heaps are stored in arrays", "source": "S1", "quote": "stored in arrays"}]])
        self.assertEqual((r["claims"][0]["kind"], r["claims"][0]["supported"]), ("knowledge", False))
        self.assertIn("These facts are not stated in the sources", Q.problems(r)[0])

    def test_the_rod_cutting_draft_now_passes_and_the_unstated_complexity_still_fails(self):
        # ALGO_06 hard, 20 Oct: draft 3 was rejected because the student's own
        # design steps ("keep the larger price in P[length]") are in no book
        sources = number_sources([
            T.source("APS_020_01", "main", "ALGO_06", "Compute the value of an optimal solution, "
                                                      "typically in a bottom-up fashion."),
            T.source("APS_020_09", "main", "ALGO_06", "rn = max\n1≤i≤n (pi + rn−i) . with r0 = 0"),
            T.source("APS_011_09", "blend", "DS_02", "the array C[0 .. k] provides temporary working storage.")])
        design = [W("Initialise P[1..k] to 0", ["S3"]), W("Scan A once", ["S3"]),
                  W("If price > P[length], set P[length] = price", ["S3"])]
        dp = [("r_n = max over i of (p_i + r_(n-i))", "S2", "rn = max 1≤i≤n (pi + rn−i)"),
              W("Loop i = 1..n, inner j = 1..i, store r[i]; the answer is r[n]", ["S2", "S1"])]
        report = Q.parse_report(check_reply([design, dp]), sources=sources, blend=None, outline_lines=2)
        self.assertEqual(Q.problems(report), [])
        complexity = ("Counting sort runs in O(m + n) time", None, "")
        report = Q.parse_report(check_reply([design + [complexity], dp]), sources=sources, blend=None,
                                outline_lines=2)
        found = Q.problems(report)
        self.assertEqual(len(found), 1)
        self.assertIn('"Counting sort runs in O(m + n) time" (outline line 1: in no source)', found[0])

    def test_both_kinds_are_explained_to_the_checker_and_the_writer(self):
        checker_prompt = Q.build_messages(draft_state())[1]["content"]
        for words in ("- KNOWLEDGE: something a student must have learned", "- WORKING: what the student "
                      "produces", "When unsure, it is knowledge.", '"kind": "working", "builds_on"'):
            self.assertIn(words, checker_prompt)
        self.assertIn("Steps the student works out", T.D.SYSTEM_PROMPT)

    def test_the_printout_shows_what_a_working_step_builds_on(self):
        report = self.report([[W("Insert 4 first", ["S1"])], [W("Put 2 right of 4", ["S1"], follows=False)]])
        report.update(problems=Q.problems(report), passed=False, model="m")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_checks({"quality_reports": [report], "drafts": [], "draft_attempts": 1})
        out = printed.getvalue()
        self.assertIn("ok  (line 1) Insert 4 first\n        [working, builds on S1]", out)
        self.assertIn("NO  (line 2) Put 2 right of 4\n        [working, builds on S1, does NOT follow]", out)


# ---------------------------------------------------------------- balanced grounding
def K(claim, source=None, quote="", correct=True, standard=True):
    """A KNOWLEDGE claim, with the checker's correct / standard judgement."""
    return {"claim": claim, "kind": "knowledge", "source": source, "quote": quote,
            "correct": correct, "standard": standard}


class TestBalancedGrounding(unittest.TestCase):
    """GROUNDING=balanced: strict on the question, light on the answer outline."""

    def report(self, lines, question=(), mode="balanced"):
        return Q.parse_report(check_reply(lines, question=question), sources=SOURCES, blend=None,
                              outline_lines=len(lines), mode=mode)

    def statuses(self, report):
        return [c["status"] for c in report["claims"]]

    def test_standard_facts_the_sources_do_not_state_pass_in_the_answer(self):
        # DS_03 draft 1 and ALGO_06 draft 2 (20 Oct) were rejected for these lines
        lines = [[TWO_CHILDREN],
                 [K("Parent of node i is at floor(i/2), children at 2i and 2i+1"),
                  K("Counting sort runs in O(n + k) time"), {"claim": "r[0] = 0", "kind": "knowledge"}]]
        balanced, strict = self.report(lines), self.report(lines, mode="strict")
        self.assertEqual(self.statuses(balanced), ["grounded", "model knowledge", "model knowledge",
                                                   "model knowledge"])
        self.assertEqual(Q.problems(balanced), [])
        self.assertEqual(balanced["counts"], {"grounded": 1, "model knowledge": 3, "working": 0})
        self.assertIn("These facts are not stated in the sources", Q.problems(strict)[0])

    def test_a_wrong_fact_fails_anywhere(self):
        r = self.report([[TWO_CHILDREN], [K("Heapify runs in O(1) time", correct=False)]])
        self.assertEqual(Q.problems(r), ['These facts are wrong: "Heapify runs in O(1) time" (outline line 2). '
                                         'Correct them.'])
        r = self.report([[TWO_CHILDREN]], question=[K("Inorder visits the root first", correct=False)])
        self.assertIn('"Inorder visits the root first" (in the question)', Q.problems(r)[0])

    def test_the_question_text_must_be_confirmed_correct_and_at_course_level(self):
        beyond = K("Fibonacci heaps decrease a key in O(1) amortized time", standard=False)
        unsure = {"claim": "A treap is a BST", "kind": "knowledge", "source": None, "quote": ""}
        r = self.report([[TWO_CHILDREN]], question=[beyond, unsure, K("A heap is a complete binary tree")])
        self.assertEqual(self.statuses(r)[:3], ["beyond level", "unconfirmed", "model knowledge"])
        text = " ".join(Q.problems(r))
        self.assertIn('go beyond what this course teaches at this semester: "Fibonacci heaps decrease', text)
        self.assertIn('neither quoted from a source nor confirmed correct and standard for this course: '
                      '"A treap is a BST" (in the question)', text)

    def test_in_the_answer_outline_only_a_wrong_step_fails(self):
        r = self.report([[TWO_CHILDREN], [W("Insert 4, then 2 and 6", []), W("Put 2 right of 4", [], follows=False)]])
        self.assertEqual(self.statuses(r), ["grounded", "working", "wrong step"])
        self.assertIn('"Put 2 right of 4" (outline line 2: does not follow)', Q.problems(r)[0])

    def test_the_question_must_be_built_on_a_main_topic_source(self):
        no_quote = self.report([[K("A heap is a complete binary tree")]])
        only_blend = self.report([[POINTER]])                               # S3 is about the prerequisite
        for r in (no_quote, only_blend):
            self.assertFalse(r["anchored"])
            self.assertIn("not built on the course material: none of its facts is quoted from a source "
                          "about the main topic (S1, S2)", " ".join(Q.problems(r)))
        self.assertTrue(self.report([[TWO_CHILDREN, K("A heap is a complete binary tree")]])["anchored"])

    def test_the_checker_is_told_the_course_level_and_what_it_may_accept(self):
        balanced = Q.build_messages(draft_state())
        strict = Q.build_messages(draft_state(), grounding="strict")
        self.assertIn('COURSE: semester 3, topic "Trees"', balanced[1]["content"])
        self.assertIn("taught in a typical semester-3 course on this topic", balanced[1]["content"])
        self.assertIn("may use correct, standard facts that no source states", balanced[1]["content"])
        self.assertNotIn("may use correct, standard facts", strict[1]["content"])
        self.assertIn("Be strict about the QUESTION", balanced[0]["content"])
        self.assertEqual(strict[0]["content"], Q.SYSTEM_PROMPT)

    def test_the_mode_must_be_known(self):
        with self.assertRaises(ValueError):
            Q.make_quality_check_node(None, grounding="loose")

    def test_an_accepted_question_lists_the_facts_to_review(self):
        client, _, _ = T.make_client(DRAFT_1, check_reply([
            [("A node has two children", "S1", "node has left and right children")],
            [K("A pointer to a node holds its address"), POINTER]]))
        r = run(client)
        self.assertTrue(accepted(r))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_checks(r)
        out = printed.getvalue()
        self.assertIn("ok  (line 2) A pointer to a node holds its address\n"
                      "        [model knowledge: not in the sources -- check when reviewing]", out)
        self.assertIn("facts: 2 grounded, 1 model knowledge, 0 working steps  (balanced grounding)", out)
        self.assertIn("Facts from the model's own knowledge:\n  - A pointer to a node holds its address", out)

    def test_strict_grounding_can_still_be_chosen_for_the_whole_loop(self):
        bad_quote = check_reply([[("A node has two children", "S1", "node has left and right children")],
                                 [("Postorder visits left, right, root", "S2", "postorder visits left right root")]])
        client, chat, _ = T.make_client(DRAFT_1, bad_quote)
        self.assertTrue(accepted(run(client)))                                    # balanced: fine
        client, chat, _ = T.make_client(DRAFT_1, bad_quote)
        self.assertFalse(accepted(run(client, grounding="strict", max_drafts=1)))  # strict: no quote, no pass
        self.assertIn("Use ONLY facts from the numbered SOURCES", chat.sent[0]["messages"][0]["content"])


# ---------------------------------------------------------------- the loop
def run(llm, checker=None, **options):
    res = Resources(driver=P.FakeDriver(P.GRAPH), grounding_collections=[P.BOOK],
                    pyq_collection=P.PYQS, embed_fn=P.fake_embed, llm=llm, checker=checker)
    return build_graph(res, **options).invoke({"current_topic_id": "DS_03", "target_difficulty": "hard",
                                               "target_marks": 5})


# P.BOOK: S1 = b1 "A binary tree node has left and right children",
#         S2 = b2 "Tree traversal: inorder, preorder", S3 = b3 "A pointer stores an address" (blend)
LOOP_GOOD = check_reply([[("A node has two children", "S1", "node has left and right children")],
                         [("A pointer stores an address", "S3", "A pointer stores an address")]])
LOOP_BAD = check_reply([[("A node has two children", "S1", "node has left and right children")],
                        [{"claim": "Postorder visits the root first", "kind": "knowledge", "source": None,
                          "quote": "", "correct": False, "standard": True}]],
                       feedback="Use only the traversals S2 names.")
DRAFT_1 = T.reply(sources_used=["S1", "S3"], question="Question one about postorder.")
DRAFT_2 = T.reply(sources_used=["S1", "S3"], question="Question two about inorder.")


class TestDraftCheckLoop(unittest.TestCase):
    def test_a_failed_draft_is_rewritten_with_the_feedback_and_then_accepted(self):
        client, chat, _ = T.make_client(DRAFT_1, LOOP_BAD, DRAFT_2, LOOP_GOOD)
        r = run(client)
        self.assertTrue(accepted(r))
        self.assertEqual(r["draft"]["question"], "Question two about inorder.")
        self.assertEqual(r["draft_attempts"], 2)
        self.assertEqual([q["passed"] for q in r["quality_reports"]], [False, True])
        rewrite = chat.sent[2]["messages"][1]["content"]                # the second draft's prompt
        self.assertIn("YOUR PREVIOUS DRAFT WAS REJECTED", rewrite)
        self.assertIn('These facts are wrong: "Postorder visits the root first" (outline line 2)', rewrite)
        self.assertIn("Previous draft:\nQuestion one about postorder.", rewrite)
        self.assertEqual(len(chat.sent), 4)
        self.assertEqual([d["question"] for d in r["drafts"]],
                         ["Question one about postorder.", "Question two about inorder."])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_checks(r)
        out = printed.getvalue()
        self.assertIn("----- draft 1 (rejected) -----\n  Question one about postorder.", out)
        self.assertIn("Check 1 (fake-model): FAIL", out)
        self.assertIn("NO  (line 2) Postorder visits the root first\n        [wrong]", out)
        self.assertNotIn("draft 2 (rejected)", out)
        self.assertIn("ACCEPTED after 2 draft(s)\nThe answer outline is a draft answer key: check it before use.", out)

    def test_after_max_drafts_failures_the_question_is_rejected(self):
        client, chat, _ = T.make_client(DRAFT_1, LOOP_BAD, DRAFT_1, LOOP_BAD, DRAFT_1, LOOP_BAD)
        r = run(client)
        self.assertFalse(accepted(r))
        self.assertEqual((r["draft_attempts"], len(r["quality_reports"]), len(chat.sent)), (3, 3, 6))
        client, chat, _ = T.make_client(DRAFT_1, LOOP_BAD)
        self.assertEqual(run(client, max_drafts=1)["draft_attempts"], 1)

    def test_a_checker_failure_stops_the_loop(self):
        client, chat, _ = T.make_client(DRAFT_1, "no", "no", "no")
        r = run(client)
        self.assertFalse(accepted(r))
        self.assertEqual((r["draft_attempts"], len(chat.sent)), (1, 4))

    def test_a_second_model_can_do_the_checking(self):
        writer, written, _ = T.make_client(DRAFT_1, DRAFT_2)
        checker, checked, _ = T.make_client(LOOP_BAD, LOOP_GOOD)
        r = run(writer, checker)
        self.assertTrue(accepted(r))
        self.assertEqual((len(written.sent), len(checked.sent)), (2, 2))

    def test_without_the_check_one_draft_is_made(self):
        client, chat, _ = T.make_client(DRAFT_1)
        r = run(client, check=False)
        self.assertEqual((r["draft"]["question"], len(chat.sent)), ("Question one about postorder.", 1))
        self.assertNotIn("quality", r)

    def test_a_topic_without_text_is_neither_drafted_nor_checked(self):
        client, chat, _ = T.make_client()
        res = Resources(driver=P.FakeDriver(P.GRAPH), grounding_collections=[P.BOOK],
                        pyq_collection=P.PYQS, embed_fn=P.fake_embed, llm=client)
        r = build_graph(res).invoke({"current_topic_id": "DS_07", "target_difficulty": "easy",
                                     "target_marks": 2})
        self.assertFalse(accepted(r))
        self.assertEqual(chat.sent, [])


if __name__ == "__main__":
    unittest.main()

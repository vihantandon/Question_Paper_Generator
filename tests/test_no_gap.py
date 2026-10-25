"""
A paper should not get a gap when the writer cannot link the prerequisite or
breaks the reply format (25 Oct: DS_03 + Linear DS failed three runs in a row).

    python3 -m unittest tests.test_no_gap -v      (from the repo root)
"""
import contextlib
import io
import unittest

import tests.test_draft as T
import tests.test_quality_check as QT
from pipeline.graph import accepted, print_draft
from pipeline.nodes import draft as D
from pipeline.nodes import quality_check as Q

NO_BLEND = T.reply(sources_used=["S1"], blend_used="")              # ignores S3, the prerequisite
PLAIN = T.reply(sources_used=["S1", "S2"], blend_used="", question="A question on trees alone.")


class TestTheBlendCanBeDropped(unittest.TestCase):
    def test_after_three_tries_a_question_on_the_main_topic_alone(self):
        client, chat, _ = T.make_client(NO_BLEND, NO_BLEND, NO_BLEND, PLAIN)
        out = D.make_draft_node(client)(T.state())
        self.assertEqual(out["draft"]["question"], "A question on trees alone.")
        self.assertEqual(out["draft"]["blend_dropped"], 'no link to "Pointers": the writer did not use its '
                                                        'sources, so this is a question on the main topic alone')
        plain_prompt = chat.sent[3]["messages"][1]["content"]
        self.assertNotIn("PREREQUISITE TO BLEND IN", plain_prompt)
        self.assertNotIn("[S3]", plain_prompt)                         # the prerequisite's source is left out
        self.assertIn("The question should test this topic on its own.", plain_prompt)
        self.assertIn("[S2]", plain_prompt)                            # the main sources keep their numbers

    def test_the_check_does_not_ask_for_the_dropped_prerequisite(self):
        draft = dict(QT.draft_state()["draft"], blend_dropped="no link")
        client, chat, _ = T.make_client(dict(QT.GOOD, blend_needed=None))
        out = Q.make_quality_check_node(client)(QT.draft_state(draft=draft))
        self.assertNotIn("PREREQUISITE THE QUESTION MUST NEED", chat.sent[0]["messages"][1]["content"])
        self.assertTrue(out["quality_passed"])

    def test_other_problems_do_not_drop_the_blend(self):
        client, chat, _ = T.make_client("no", "no", "no")
        out = D.make_draft_node(client)(T.state())
        self.assertIsNone(out["draft"])
        self.assertEqual(len(chat.sent), 3)
        self.assertIn("Your last reply could not be used", out["revision_feedback"])

    def test_the_printout_says_so(self):
        printed = io.StringIO()
        draft = D.parse_draft(PLAIN, marks=5, sources=T.SOURCES)
        draft["blend_dropped"] = "no link to Pointers"
        with contextlib.redirect_stdout(printed):
            print_draft({"draft": draft, "draft_attempts": 1})
        self.assertIn("Blend: dropped -- no link to Pointers", printed.getvalue())


class TestUnusableRepliesCostOneDraft(unittest.TestCase):
    def test_a_fresh_draft_is_started(self):
        client, chat, _ = T.make_client("no", "no", "no", QT.DRAFT_1, QT.LOOP_GOOD)
        r = QT.run(client)
        self.assertTrue(accepted(r))
        self.assertEqual((r["draft_attempts"], len(chat.sent)), (2, 5))
        self.assertIn("Your last reply could not be used", chat.sent[3]["messages"][1]["content"])

    def test_at_most_max_drafts(self):
        client, chat, _ = T.make_client(*["no"] * 6)
        r = QT.run(client, max_drafts=2)
        self.assertFalse(accepted(r))
        self.assertEqual((r["draft_attempts"], len(chat.sent)), (2, 6))


class TestInAPaper(unittest.TestCase):
    def test_the_dropped_link_is_a_note_on_the_question(self):
        import tests.test_paper as TP
        p = TP.Paper(NO_BLEND, NO_BLEND, NO_BLEND, PLAIN, QT.LOOP_GOOD, TP.CO4,
                     TP.SECOND, QT.LOOP_GOOD, TP.CO4)
        slots, stopped = p.make()
        self.assertIsNone(stopped)
        self.assertEqual([s["status"] for s in slots.values()], ["done", "done"])
        self.assertIn('no link to "Pointers"', " ".join(slots[1]["notes"]))


if __name__ == "__main__":
    unittest.main()

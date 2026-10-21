"""
Redoing the sums a draft writes down (pipeline/arithmetic.py), and the
quality check failing a draft whose sums are wrong.

    python3 -m unittest tests.test_arithmetic -v      (from the repo root)
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_draft as T  # noqa: E402
from pipeline.arithmetic import check_arithmetic, evaluate  # noqa: E402
from pipeline.graph import print_check  # noqa: E402
from pipeline.nodes import quality_check as Q  # noqa: E402
from tests.test_quality_check import GOOD, draft_state  # noqa: E402

# line 7 of the answer outline of the ALGO_06 question accepted on 20 Oct
# (‑ is the non-breaking hyphen the model wrote for minus)
ROD_LINE = ("For the instance, after sorting price = [2,5,8,9,12,17]; compute R[1]=2, "
            "R[2]=max(5,2+2‑2)=5, R[3]=max(8,2+5‑2,5+2‑2)=8, R[4]=max(9,2+8‑2,5+5‑2,8+2‑2)=11, "
            "R[5]=max(12,2+11‑2,5+8‑2,8+5‑2,9+2‑2)=13, R[6]=max(17,2+13‑2,5+11‑2,8+8‑2,9+5‑2,12+2‑2)=17. "
            "Hence maximum net revenue = 17.")


class TestTheRodCuttingAnswerKey(unittest.TestCase):
    def test_both_wrong_sums_are_found_and_the_right_ones_pass(self):
        checked, wrong = check_arithmetic(ROD_LINE)
        self.assertEqual(checked, 5)                     # R[2] .. R[6]
        self.assertEqual(wrong, [
            {"text": "max(9,2+8-2,5+5-2,8+2-2) = 11", "note": "it is 9"},
            {"text": "max(12,2+11-2,5+8-2,8+5-2,9+2-2) = 13", "note": "it is 12"}])

    def test_sums_between_arrows_are_worked_out(self):
        # the rod-cutting table of the next run: "... = 7+2-1=8 -> R[5]=8"
        line = "j=5: L=1 → 2 + R[4] – c = 2+7‑1=8; L=4 → 7 + R[1] – c = 7+2‑1=8 → R[5]=8"
        self.assertEqual(check_arithmetic(line), (2, []))
        self.assertEqual(check_arithmetic("2+7-1=9 -> R[5]")[1], [{"text": "2+7-1 = 9", "note": "it is 8"}])

    def test_the_corrected_line_passes(self):
        fixed = ROD_LINE.replace("8+2‑2)=11", "8+2‑2)=9").replace("9+2‑2)=13", "9+2‑2)=12")
        self.assertEqual(check_arithmetic(fixed), (5, []))


class TestWhatIsWorkedOut(unittest.TestCase):
    def test_right_calculations(self):
        for text in ("total cost = 5 + 8 = 13.", "h(23) = 23 % 7 = 2", "2^10 = 1024", "2 × 3 = 6",
                     "sum = 3+4+5 = 12, avg = 12/3 = 4", "8 − 2 = 6", "1. 2+3 = 5", "- 2+3 = 5",
                     "min(4, 7) + 1 = 5", "0.1+0.2 = 0.3"):
            checked, wrong = check_arithmetic(text)
            self.assertGreater(checked, 0, text)
            self.assertEqual(wrong, [], text)

    def test_wrong_calculations(self):
        for text, note in (("total cost = 5 + 8 = 12.", "it is 13"),
                           ("h(23) = 23 % 7 = 3", "it is 2"),
                           ("a = b + 1 = 2+2 = 5", "it is 4"),
                           ("2+3 = 1+5", "the two sides are 5 and 6")):
            checked, wrong = check_arithmetic(text)
            self.assertEqual([w["note"] for w in wrong], [note], text)

    def test_division_may_round_down_or_up_and_decimals_may_be_rounded(self):
        for text in ("mid = (0+9)/2 = 4", "-7/2 = -3", "⌊9/2⌋ = 4", "⌈9/2⌉ = 5", "10/3 = 3.33",
                     "0.333 = 1/3", "load factor = 7/10 = 0.7"):
            self.assertEqual(check_arithmetic(text)[1], [], text)
        self.assertEqual(len(check_arithmetic("10/3 = 3.5")[1]), 1)
        self.assertEqual(len(check_arithmetic("⌊9/2⌋ = 5")[1]), 1)

    def test_what_is_not_a_calculation_is_left_alone(self):
        # names, words and code around the numbers: misreading them would reject good questions
        for text in ("n = 6, c = 2, unsorted pairs = {(3,8),(1,2)}", "R[0]=0", "x1 = 5", "x_1 = 5",
                     "3 log 8 = 9", "multiply 3 by 4 = 12", "T(n) = 2T(n/2) + n = O(n log n)",
                     "i==j?0:c", "if (a <= 3) b = 4", "for (i = 0; i < n; i++)", "x := 3+4 = 8",
                     "= 2n", "1,000 = 1000", "values = 2, 5, 8", "50% = 0.5", "5 = 5",
                     "R[j] = max_{1≤i≤j} (price[i] + R[j‑i] - (i==j?0:c))", "log2(8) = 4",
                     "4(4+1)/2 = 11", "Hence 2+3 = 6", "cost is 5+8 = 12", r"$\frac{3}{4} = 0.8$"):
            self.assertEqual(check_arithmetic(text), (0, []), text)

    def test_evaluate_allows_only_numbers_operators_and_four_functions(self):
        self.assertEqual(evaluate("max(9, 2+8-2) * 2"), 18)
        self.assertEqual(evaluate("2^3"), 8)
        for text in ("__import__('os')", "len([1])", "9 ** 999999", "1/0", "(1, 2)", "abc", ""):
            self.assertIsNone(evaluate(text), text)


class TestTheQualityCheckRedoesTheSums(unittest.TestCase):
    def test_a_wrong_sum_fails_the_draft_even_when_the_checker_saw_nothing(self):
        state = draft_state()
        state["draft"] = dict(state["draft"], answer_outline=["root 4", "R[4]=max(9,2+8-2)=11"])
        client, chat, _ = T.make_client(GOOD)
        out = Q.make_quality_check_node(client)(state)
        self.assertFalse(out["quality_passed"])
        self.assertEqual(out["quality"]["arithmetic"],
                         {"checked": 1, "wrong": [{"text": "max(9,2+8-2) = 11", "note": "it is 9",
                                                   "where": "outline line 2"}]})
        self.assertIn('These calculations are wrong: "max(9,2+8-2) = 11" (outline line 2: it is 9). '
                      "Work every number out again", out["revision_feedback"])
        self.assertIn("redo every calculation yourself", chat.sent[0]["messages"][1]["content"])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_check(1, out["quality"])
        self.assertIn("calculations: 1 worked out again, 1 wrong", printed.getvalue())

    def test_a_sum_in_the_question_text_counts_too(self):
        state = draft_state()
        state["draft"] = dict(state["draft"], question="Total size: 3 + 4 = 8. Insert the keys.")
        client, _, _ = T.make_client(GOOD)
        out = Q.make_quality_check_node(client)(state)
        self.assertEqual(out["quality"]["arithmetic"]["wrong"][0]["where"], "in the question")

    def test_right_sums_change_nothing(self):
        state = draft_state()
        state["draft"] = dict(state["draft"], answer_outline=["root 4", "R[4]=max(9,2+8-2)=9"])
        client, _, _ = T.make_client(GOOD)
        out = Q.make_quality_check_node(client)(state)
        self.assertTrue(out["quality_passed"])
        self.assertEqual(out["quality"]["arithmetic"], {"checked": 1, "wrong": []})


if __name__ == "__main__":
    unittest.main()

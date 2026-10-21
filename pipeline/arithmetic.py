"""
pipeline/arithmetic.py -- redo the sums a draft writes down.

An accepted ALGO_06 answer key said

    R[4]=max(9,2+8-2,5+5-2,8+2-2)=11, R[5]=max(12,2+11-2,5+8-2,8+5-2,9+2-2)=13

The first max is 9 and the second 12. The final answer (17) was right by luck,
and the checking model called the line "working, follows" without redoing the
sums. A language model is unreliable at arithmetic; Python is not. So this
file finds every written-out calculation -- "a = b", "a = b = c" -- and works
both sides out again:

    check_arithmetic("R[4]=max(9,2+8-2)=11")  ->  1 checked, 1 wrong (it is 9)

WHAT COUNTS AS A CALCULATION
    numbers, + - * / % ^ ( ), and max / min / floor / ceil (also the
    ⌊ ⌋ ⌈ ⌉ signs). "×", "÷" and the minus-sign look-alikes are read too.
    Both sides of an "=" must be such an expression, and at least one side
    must actually compute something (not just "5 = 5").

WHAT IS LEFT ALONE (it is easy to misread, and a false alarm would reject a
good question)
    * anything joined to words or names: "n = 6", "R[2] = 5", "x1 = 4",
      "3 log 8 = 9", "multiply 3 by 4 = 12", "= 2n"
    * ==, <=, >=, !=, := (comparisons and pseudocode assignment)
    * lists and tuples ("= 2, 5, 8"), ranges, LaTeX, anything that does not parse

Integer division: "(0+9)/2 = 4" is fine -- an algorithm (and C) rounds down,
and "⌈7/2⌉"-style rounding up is fine too. A stated decimal is fine when it is
the exact value rounded to the digits shown ("10/3 = 3.33").
"""
import ast
import math
import operator
import re

# look-alikes -> plain ASCII
_SYMBOLS = str.maketrans({
    "−": "-", "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "×": "*", "·": "*", "∗": "*", "⋅": "*",
    "÷": "/", "∕": "/",
})
_NOT_EQUALS = re.compile(r"==|<=|>=|!=|:=|=>|=<")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+", re.MULTILINE)

# a run of calculation characters: digits, operators, brackets, commas, spaces,
# and the four function names
_RUN = r"(?:max|min|floor|ceil|[0-9.+\-*/%^(),\s])+"
_RUN_AT_END = re.compile(_RUN + r"$")
_RUN_AT_START = re.compile(_RUN)

# what may stand right before / after a calculation (spaces skipped): the start
# or end of the text, or punctuation. A letter, digit, ] or _ there means the
# numbers belong to something else ("x1", "R[2]", "3 log 8", "2n").
_OK_BEFORE = set(",;:(\n$\u2192\u21d2")        # also the arrows -> and =>
_OK_AFTER = set(",;:.)\n$\u2192\u21d2")

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_FUNCS = {"max": max, "min": min, "floor": math.floor, "ceil": math.ceil}


class _NotACalculation(Exception):
    pass


def _value(node):
    """Work out a parsed expression, allowing only numbers, + - * / % ** and
    the four functions."""
    if isinstance(node, ast.Expression):
        return _value(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 10 ** 6):
            raise _NotACalculation("too big")
        return _BIN[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and node.args and not node.keywords):
        args = [_value(a) for a in node.args]
        if node.func.id in ("floor", "ceil"):
            if len(args) != 1:
                raise _NotACalculation("floor/ceil take one number")
            return _FUNCS[node.func.id](args[0])
        return _FUNCS[node.func.id](args)
    raise _NotACalculation(type(node).__name__)


def evaluate(expr):
    """The value of a calculation written as text, or None when it is not
    one (it does not parse, or uses anything but numbers, operators and
    max / min / floor / ceil)."""
    if not re.search(r"\d", expr or ""):
        return None
    try:
        value = _value(ast.parse(expr.strip().replace("^", "**"), mode="eval"))
    except (SyntaxError, _NotACalculation, ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _is_bare_number(expr):
    return re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", expr) is not None


def _computes(expr):
    """Does this side work something out (not just state a number)?"""
    return not _is_bare_number(expr)


def _balance(expr, opening_side):
    """Drop brackets that belong to the sentence around the calculation:
    "(total = 5+3)" gives the run "5+3)"."""
    if opening_side:
        while expr.startswith("(") and expr.count("(") > expr.count(")"):
            expr = expr[1:].lstrip()
    else:
        while expr.endswith(")") and expr.count(")") > expr.count("("):
            expr = expr[:-1].rstrip()
    return expr


def _left_side(segment):
    """The calculation at the end of the text before an "=", or None."""
    m = _RUN_AT_END.search(segment)
    if not m:
        return None
    run = m.group().rstrip()
    expr = run.lstrip(" \t\n,;")
    if expr.startswith(".") and not expr[1:2].isdigit():     # the full stop of a sentence
        expr = expr[1:].lstrip()
    expr = _balance(expr, opening_side=True)
    before = segment[:m.start() + len(run) - len(expr)].rstrip(" \t")
    if before and before[-1] not in _OK_BEFORE:
        return None
    return expr or None


def _right_side(segment):
    """The calculation at the start of the text after an "=", or None."""
    m = _RUN_AT_START.match(segment)
    if not m:
        return None
    run = m.group()
    lead = len(run) - len(run.lstrip())
    expr = run.strip().rstrip(" \t\n,;")
    while expr.endswith("."):                    # the full stop of a sentence
        expr = expr[:-1].rstrip(" \t\n,;")
    expr = _balance(expr, opening_side=False)
    after = segment[lead + len(expr):].lstrip(" \t")
    if after and after[0] not in _OK_AFTER:
        return None
    return expr or None


def _agrees(left, right, a, b):
    if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
        return True
    for stated, computed, other in ((right, a, left), (left, b, right)):
        if not _is_bare_number(stated):
            continue
        value = float(stated)
        decimals = len(stated.strip().partition(".")[2])
        if decimals and abs(computed - value) <= 0.5 * 10 ** -decimals + 1e-9:
            return True                       # rounded to the digits shown
        if "/" in other and not decimals and value in (math.floor(computed), math.ceil(computed)):
            return True                       # integer division
    if "/" in left + right and _computes(left) and _computes(right):
        return math.floor(a) == math.floor(b) or math.ceil(a) == math.ceil(b)
    return False


def _show(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.4g}"


def check_arithmetic(text):
    """Find the calculations in `text` and redo them.

    Returns (checked, wrong): how many "a = b" calculations were worked out,
    and for each wrong one {"text": "max(9,2+8-2) = 11", "note": "it is 9"}."""
    text = _BULLET.sub("", (text or "").translate(_SYMBOLS)).replace("->", "\u2192")
    text = text.replace("\u230a", "floor(").replace("\u230b", ")")
    text = text.replace("\u2308", "ceil(").replace("\u2309", ")")
    text = _NOT_EQUALS.sub(" \x00 ", text)
    checked, wrong = 0, []
    for line in text.split("\n"):
        segments = line.split("=")
        for i in range(len(segments) - 1):
            left, right = _left_side(segments[i]), _right_side(segments[i + 1])
            if not left or not right or not (_computes(left) or _computes(right)):
                continue
            a, b = evaluate(left), evaluate(right)
            if a is None or b is None:
                continue
            checked += 1
            if _agrees(left, right, a, b):
                continue
            if _computes(left) and _computes(right):
                note = f"the two sides are {_show(a)} and {_show(b)}"
            else:
                note = f"it is {_show(a if _computes(left) else b)}"
            wrong.append({"text": f"{_tidy(left)} = {_tidy(right)}", "note": note})
    return checked, wrong


def _tidy(expr):
    return re.sub(r"\s+", " ", expr).strip()

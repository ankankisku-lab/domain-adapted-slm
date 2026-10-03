"""Executor for the FinQA program DSL, plus answer-agreement checks.

Semantics follow the FinQA reference implementation (Chen et al., 2021): numbers may carry "%" (divided by 100),
"const_m1" is -1, "#k" refers to step k's result, table ops aggregate the numeric cells of the row whose first cell
matches the argument, and greater() returns "yes"/"no".
"""

import math
import re

Result = float | str | None  # None = program could not be executed


def str_to_num(text: str) -> float | None:
    text = text.replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        pass
    if "%" in text:
        try:
            return float(text.replace("%", "")) / 100.0
        except ValueError:
            return None
    if text.startswith("const_"):
        value = text[len("const_"):]
        return -1.0 if value == "m1" else float(value)
    return None


def parse_program(program: str) -> list[tuple[str, str, str]] | None:
    """'divide(100, 100), divide(3.8, #0)' -> [('divide', '100', '100'), ('divide', '3.8', '#0')].

    Row names may contain parentheses and commas, so parenthesis depth is tracked and each step is split on its
    last top-level ", " (every op takes exactly two arguments).
    """
    steps, i, n = [], 0, len(program)
    while i < n:
        while i < n and program[i] in " ,":
            i += 1
        if i >= n:
            break
        open_idx = program.find("(", i)
        if open_idx == -1:
            return None
        op = program[i:open_idx].strip()
        depth, j = 1, open_idx + 1
        while j < n and depth:
            depth += {"(": 1, ")": -1}.get(program[j], 0)
            j += 1
        if depth:
            return None
        args = program[open_idx + 1 : j - 1].rsplit(", ", 1)
        if len(args) != 2 or not op.isidentifier():
            return None
        steps.append((op, args[0].strip(), args[1].strip()))
        i = j
    return steps or None


def row_values(table: list[list[str]], row_name: str) -> list[float] | None:
    for row in table:
        if row and row[0].strip() == row_name.strip():
            values = []
            for cell in row[1:]:
                num = str_to_num(cell.replace("$", "").split("(")[0].strip())
                if num is None:
                    return None
                values.append(num)
            return values or None
    return None


def execute(program: str, table: list[list[str]]) -> Result:
    results = execute_steps(program, table)
    return results[-1] if results else None


def execute_steps(program: str, table: list[list[str]]) -> list[float | str] | None:
    """Result of every step, in order, or None if the program can't be executed."""
    steps = parse_program(program)
    if steps is None:
        return None
    results: list[float | str] = []

    def arg(a: str) -> float | None:
        if a.startswith("#"):
            k = int(a[1:]) if a[1:].isdigit() else -1
            return results[k] if 0 <= k < len(results) and isinstance(results[k], float) else None
        return str_to_num(a)

    for op, a1, a2 in steps:
        if op.startswith("table_"):
            values = row_values(table, a1)
            if values is None:
                return None
            res = {"table_max": max, "table_min": min, "table_sum": sum,
                   "table_average": lambda v: sum(v) / len(v)}.get(op, lambda v: None)(values)
        else:
            x, y = arg(a1), arg(a2)
            if x is None or y is None:
                return None
            try:
                res = {"add": lambda: x + y, "subtract": lambda: x - y, "multiply": lambda: x * y,
                       "divide": lambda: x / y, "exp": lambda: x ** y,
                       "greater": lambda: "yes" if x > y else "no"}[op]()
            except (KeyError, ZeroDivisionError, OverflowError):
                return None
        if res is None or (isinstance(res, float) and not math.isfinite(res)):
            return None
        results.append(res)
    return results


def same_result(a: Result, b: Result, rel: float = 1e-4, abs_tol: float = 1e-5) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip().lower() == str(b).strip().lower()
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


_ANSWER_NUM = re.compile(r"^\(?(-?\$?\s*-?[\d,]*\.?\d+)\)?\s*(%|million|billion|thousand|m|b|k)?$", re.I)


def parse_answer_text(text: str) -> tuple[float, int, bool] | str | None:
    """Annotator answer -> (value, decimals shown, has_percent), "yes"/"no", or None if unparseable."""
    t = text.strip().lower().rstrip(".")
    if t in ("yes", "no"):
        return t
    m = _ANSWER_NUM.match(t.replace(" ", ""))
    if not m:
        return None
    num = m.group(1).replace("$", "").replace(",", "")
    try:
        value = float(num)
    except ValueError:
        return None
    if t.startswith("(") and t.endswith(")"):
        value = -abs(value)  # accounting negative
    decimals = len(num.split(".")[1]) if "." in num else 0
    return value, decimals, m.group(2) == "%"


_ANY_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _matches(value: float, decimals: int, exe_ans: float) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - c) <= max(tol, 0.005 * abs(c)) for c in (exe_ans, exe_ans * 100))


def answer_agrees(answer_text: str, exe_ans: Result) -> bool | None:
    """Does the annotator's answer string match the executed result, allowing for display rounding and the
    ratio-vs-percent convention (exe 0.124 == "12.4%")? None when the answer text is empty or has no number.

    Free-text answers ("increased 38.6%", "2.3:1", "$ 386797190 or $ 386.8 million") agree if any number in them
    matches.
    """
    if not answer_text.strip() or exe_ans is None:
        return None
    parsed = parse_answer_text(answer_text)
    if isinstance(parsed, str) or isinstance(exe_ans, str):
        return str(parsed) == str(exe_ans).strip().lower() if parsed is not None else None
    if parsed is not None:
        return _matches(parsed[0], parsed[1], exe_ans)
    numbers = [n.replace(",", "") for n in _ANY_NUM.findall(answer_text)]
    if not numbers:
        return None
    return any(_matches(float(n), len(n.split(".")[1]) if "." in n else 0, exe_ans) for n in numbers if n not in ("-", ""))

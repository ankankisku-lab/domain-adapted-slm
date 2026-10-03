"""Answer extraction and correctness for FinQA-style numeric answers.

One scorer is used for every model (base, SFT, ORPO, every GGUF quant) so results are comparable:
  1. extract the final answer: the last "Answer:" line if present, otherwise the last number in the output
  2. compare to the executed gold (exe_ans), allowing display rounding and the ratio/percent convention
     (gold 0.124 accepts "12.4%", "0.124", "12.4"), with a 0.5% relative floor for rounded large numbers
Yes/no questions are compared as text.
"""

import re

_ANSWER_LINE = re.compile(r"answer\s*[:：]\s*(.+)", re.I)
_NUM = re.compile(r"-?\$?\s*\(?-?\d[\d,]*\.?\d*\)?\s*%?|-?\.\d+\s*%?")
_YES_NO = re.compile(r"\b(yes|no)\b", re.I)


def extract_answer(output: str) -> str | None:
    """Final-answer text: the last 'Answer:' line, else the whole output (the last number is taken downstream)."""
    matches = _ANSWER_LINE.findall(output)
    return matches[-1].strip() if matches else (output.strip() or None)


def _parse_number(text: str) -> tuple[float, int] | None:
    """Last number in text -> (value, decimals shown). Percent signs are handled by the caller's candidates."""
    found = _NUM.findall(text.replace("−", "-"))
    for raw in reversed(found):
        s = raw.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
        negative = s.startswith("(") and s.endswith(")")
        s = s.strip("()")
        try:
            value = float(s)
        except ValueError:
            continue
        decimals = len(s.split(".")[1]) if "." in s else 0
        return (-abs(value) if negative else value), decimals
    return None


_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_PUNCT = re.compile(r"[^\w\s%.-]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _norm_text(s: str) -> str:
    s = _ARTICLES.sub(" ", _PUNCT.sub(" ", s.lower()))
    return " ".join(s.replace(",", "").split())


def _numeric(s: str) -> float | None:
    t = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    try:
        return float(t)
    except ValueError:
        return None


def _number_matches_scaled(answer: str, gold: float, scale: str) -> bool:
    """Any number in the answer equals the gold as written, as a ratio (percent scale) or fully expanded
    (thousand/million/billion), at the precision the answer states."""
    candidates = [gold]
    if scale == "percent":
        candidates.append(gold / 100)
    if scale in _SCALE:
        candidates.append(gold * _SCALE[scale])
    for raw in _NUM.findall(answer.replace("−", "-")):
        parsed = _parse_number(raw)
        if parsed and any(_matches(parsed[0], parsed[1], c) for c in candidates):
            return True
    return False


def is_correct_tatqa(output: str, record: dict) -> tuple[bool, str | None]:
    """TAT-QA: numbers (arithmetic/count, and numeric spans) compare numerically with scale-aware candidates; text
    spans are correct if the normalised gold text appears in the answer; multi-span needs every gold item."""
    answer = extract_answer(output)
    if answer is None:
        return False, None
    gold = record["exe_ans"]
    items = gold if isinstance(gold, list) else [gold]
    for item in items:
        value = item if isinstance(item, (int, float)) else _numeric(str(item))
        if value is not None:
            if not _number_matches_scaled(answer, float(value), record.get("scale", "")):
                return False, answer
        elif _norm_text(str(item)) not in _norm_text(answer):
            return False, answer
    return True, answer


def score_record(record: dict, output: str, **kwargs) -> tuple[bool, str | None]:
    """Dispatch to the right scorer for the record's dataset."""
    if record.get("dataset") == "tatqa":
        return is_correct_tatqa(output, record)
    return is_correct(output, record["exe_ans"], record["question"], **kwargs)


_DECREASE = re.compile(r"\b(decrease[sd]?|decline[sd]?|dropped|drop|fell|fall|lower|reduc\w*|down)\b", re.I)
_DIFFERENCE = re.compile(r"\bdifference\b", re.I)


# Relative tolerance on top of display rounding. 0.1% (primary) only forgives large numbers rounded to 3-4
# significant digits ("12,300" for 12,303). 0.5% (lenient, reported alongside) also accepts near-misses from rounded
# intermediate steps (45.25% for 45.10%), which are wrong at the precision the model chose to state.
REL_TOL = 0.001
REL_TOL_LENIENT = 0.005


def _matches(value: float, decimals: int, gold: float, rel_tol: float = REL_TOL) -> bool:
    tol_display = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - c) <= max(tol_display, rel_tol * abs(c)) for c in (gold, gold * 100))


def is_correct(output: str, gold: float | str, question: str = "", strict_sign: bool = False,
               rel_tol: float = REL_TOL) -> tuple[bool, str | None]:
    """Returns (correct, extracted answer text).

    Correct means equal to the gold at the precision the answer states (display rounding), with a rel_tol floor for
    large rounded numbers. Sign handling (unless strict_sign): a positive magnitude stated as a decrease
    ("declined 2.6%") matches a negative gold, and "difference between X and Y" questions accept either sign, because
    they don't say which way to subtract. The same rules apply to every model.
    """
    answer = extract_answer(output)
    if answer is None:
        return False, None
    if isinstance(gold, str):  # yes / no
        m = _YES_NO.findall(answer)
        return bool(m) and m[-1].lower() == gold.strip().lower(), answer
    parsed = _parse_number(answer)
    if parsed is None:
        return False, answer
    value, decimals = parsed
    if _matches(value, decimals, gold, rel_tol):
        return True, answer
    if strict_sign:
        return False, answer
    if gold < 0 < value and _DECREASE.search(answer) and _matches(-value, decimals, gold, rel_tol):
        return True, answer
    if _DIFFERENCE.search(question) and _matches(-value, decimals, gold, rel_tol):
        return True, answer
    return False, answer

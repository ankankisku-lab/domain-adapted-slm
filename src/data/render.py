"""Single source of truth for how a FinQA record is shown to the model (training, eval and deployment).

Context: report text before the table, the table as markdown, text after it.
Response: numbered calculation steps derived from the gold program, then one final "Answer:" line. The steps are
executed, not copied, so every intermediate number is correct by construction.
"""

from src.data.finqa_program import execute_steps, parse_answer_text, parse_program, row_values
from src.data.text_clean import clean_text

SYSTEM_PROMPT = (
    "You are a financial analyst. Answer the question using only the report excerpt provided. "
    "Show each calculation step, then give the final result on a line starting with 'Answer:'."
)

# Llama 3.x chat templates insert "Today Date: <strftime_now>" into the system turn unless date_string is passed.
# Pin it so a prompt is byte-identical across days, in training and in every evaluation.
CHAT_TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}

_OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "^"}
_TABLE_OPS = {"table_sum": "sum", "table_average": "average", "table_max": "maximum", "table_min": "minimum"}


def render_table(table: list[list[str]]) -> str:
    if not table:
        return ""
    width = max(len(row) for row in table)
    rows = [[cell.replace("|", "/") for cell in row] + [""] * (width - len(row)) for row in table]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def render_context(record: dict) -> str:
    parts = [" ".join(record["pre_text"]), render_table(record["table"]), " ".join(record["post_text"])]
    return "\n\n".join(p for p in parts if p.strip())


def render_prompt(record: dict) -> str:
    return f"{render_context(record)}\n\nQuestion: {record['question']}"


def fmt_num(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    decimals = 6 if abs(x) < 0.001 else 4
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def final_answer(record: dict) -> str:
    """Display form of the executed answer. The annotator answer (verified to agree in Phase 2) decides whether a
    ratio is shown as a percentage: exe 0.09807 with answer "9.80%" becomes "9.81%"."""
    exe = record["exe_ans"]
    if isinstance(exe, str):
        return exe
    parsed = parse_answer_text(record["answer_text"])
    shows_percent = "%" in record["answer_text"]
    if isinstance(parsed, tuple):
        value = parsed[0]
        as_ratio = abs(value - exe * 100) < abs(value - exe)  # annotator wrote exe * 100
    else:
        as_ratio = shows_percent and abs(exe) < 10
    if as_ratio:
        return f"{exe * 100:.2f}".rstrip("0").rstrip(".") + "%"
    shown = f"{exe:.2f}".rstrip("0").rstrip(".") if abs(exe) >= 1 else fmt_num(exe)
    return shown + ("%" if shows_percent else "")


def _arg_text(arg: str, results: list) -> str:
    if arg.startswith("#"):
        return fmt_num(results[int(arg[1:])])
    if arg.startswith("const_"):
        return "-1" if arg == "const_m1" else arg[len("const_"):]
    return arg


def render_response(record: dict, raw_table: list[list[str]]) -> str | None:
    """`raw_table` is the uncleaned table: program row-name arguments are written against the raw row labels."""
    steps = parse_program(record["program"])
    results = execute_steps(record["program"], raw_table)
    if not steps or not results:
        return None
    lines = []
    for i, ((op, a1, a2), res) in enumerate(zip(steps, results), 1):
        if op in _TABLE_OPS:
            values = ", ".join(fmt_num(v) for v in row_values(raw_table, a1))
            lines.append(f"{i}. {_TABLE_OPS[op]} of the '{clean_text(a1)}' row ({values}) = {fmt_num(res)}")
        elif op == "greater":
            lines.append(f"{i}. is {_arg_text(a1, results)} greater than {_arg_text(a2, results)}? {res}")
        else:
            lines.append(f"{i}. {_arg_text(a1, results)} {_OPS[op]} {_arg_text(a2, results)} = {fmt_num(res)}")
    return "Calculation:\n" + "\n".join(lines) + f"\n\nAnswer: {final_answer(record)}"

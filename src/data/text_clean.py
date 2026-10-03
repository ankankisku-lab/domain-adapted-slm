"""Regex normalization for FinQA's extraction/tokenization artifacts.

FinQA text is lowercased, whitespace-tokenized PDF extraction with duplicated annotations, e.g.
"plus 2.05% ( 2.05 % ) as of october 31 , 2009 ." and "$ -23158 ( 23158 )". Each rule is named so the cleaning
report can count how often it fired. Rules only remove duplicated annotations and spacing; they never alter digits.
"""

import re
import unicodedata
from collections import Counter

_NUM = r"\d[\d,]*(?:\.\d+)?"

# FinQA's PDF extraction replaced some Unicode punctuation with its hex code point: U+2019 (') -> "2019",
# U+2018 (') -> "2018", U+201C/U+201D (" ") -> "201c"/"201d", U+F0B7 (bullet) -> "f0b7". Only unambiguous contexts are
# decoded ("company 2019s" but never a bare "2019"). Dashes (U+2013/U+2014 -> "2013"/"2014") are left alone because
# they're indistinguishable from real years.
DECODE_RULES: list[tuple[str, re.Pattern, str]] = [
    # Must run first: in "201c2019 notes" the "c2019" would otherwise read as letter + close quote.
    ("decode_curly_double_quote", re.compile(r"\b201[cd](?=\d{4}\b|[a-z]|\s|$)"), '"'),
    ("decode_double_open_quote", re.compile(r"\b2018 2018(?=[a-z])"), '"'),
    ("decode_double_close_quote", re.compile(r"(?<=[a-z.]) 2019 2019\b"), '"'),
    ("decode_possessive", re.compile(r"(?<=\w) 2019s\b"), "'s"),
    ("decode_open_quote", re.compile(r"\b2018(?=[a-z])"), "'"),
    ("decode_close_quote", re.compile(r"(?<=[a-z])2019\b"), "'"),
    ("decode_bullet", re.compile(r"\bf0b7\b"), "-"),
]


def decode_artifacts(text: str, counts: Counter | None = None) -> str:
    for name, pattern, repl in DECODE_RULES:
        text, n = pattern.subn(repl, text)
        if counts is not None and n:
            counts[name] += n
    return text


RULES: list[tuple[str, re.Pattern, str]] = [
    # "2.05% ( 2.05 % )" -> "2.05%"   (duplicated percent annotation)
    ("dup_percent", re.compile(rf"(-?{_NUM})\s*%\s*\(\s*-?{_NUM}\s*%\s*\)"), r"\1%"),
    # "-23158 ( 23158 )" -> "-23158"  (negative with duplicated accounting-parentheses form)
    ("dup_negative", re.compile(rf"-({_NUM})\s*\(\s*\1\s*\)"), r"-\1"),
    # "( 10 ) % (  % )" -> "( 10 ) %" (empty leftover percent annotation)
    ("empty_percent_annotation", re.compile(r"%\s*\(\s*%\s*\)"), "%"),
    ("dollar_space", re.compile(r"\$\s+(?=-?\d)"), "$"),
    ("percent_space", re.compile(r"(\d)\s+%"), r"\1%"),
    ("space_before_punct", re.compile(r"\s+([,.;:!?])(?=\s|$)"), r"\1"),
    ("paren_inner_space", re.compile(r"\(\s+([^()]*?)\s+\)"), r"(\1)"),
    # "2017:." / "progress.." -> "2017:" / "progress."  (sentence-final period added after existing punctuation)
    ("double_terminal_punct", re.compile(r"([.:;])\.(?=\s|$)"), r"\1"),
    ("possessive_space", re.compile(r"\s+'(s|re|ve|ll|d|t)\b"), r"'\1"),
    ("negation_space", re.compile(r"\s+n't\b"), "n't"),
    ("multi_space", re.compile(r"[ \t]{2,}"), " "),
]

_PUNCT_ONLY = re.compile(r"^[\s.,;:!?\-–—*]*$")


def clean_text(text: str, counts: Counter | None = None) -> str:
    text = decode_artifacts(unicodedata.normalize("NFKC", text), counts)
    for name, pattern, repl in RULES:
        text, n = pattern.subn(repl, text)
        if counts is not None and n:
            counts[name] += n
    return text.strip()


def clean_sentences(sentences: list[str], counts: Counter | None = None) -> list[str]:
    """Clean each sentence and drop ones that are empty or punctuation-only after cleaning."""
    out = []
    for s in sentences:
        c = clean_text(s, counts)
        if _PUNCT_ONLY.match(c):
            if counts is not None:
                counts["dropped_empty_sentence"] += 1
            continue
        out.append(c)
    return out


def clean_table(table: list[list[str]], counts: Counter | None = None) -> list[list[str]]:
    return [[clean_text(cell, counts) for cell in row] for row in table]


def numbers_in(text: str) -> set[float]:
    """Absolute numeric values mentioned in text (used to check cleaning never loses a number)."""
    out = set()
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?|\.\d+", text):  # also ".76" (no leading zero)
        try:
            out.add(abs(float(m.replace(",", ""))))
        except ValueError:
            pass
    return out

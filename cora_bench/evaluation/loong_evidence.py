"""LOONG diagnostic: does any source line containing the gold value survive selection?

Candidate lines are unreviewed and may refer to the wrong company or period.
"""

import re
from functools import lru_cache

from cora_bench.evaluation.loong_value_match import spotlight_value_match


def normalize(text):
    return " ".join(text.lower().split())


@lru_cache(maxsize=4)
def candidate_lines(answer, context):
    result = []
    offset = 0
    for number, raw in enumerate(context.splitlines(keepends=True), 1):
        line = raw.strip()
        if line and re.search(r"[A-Za-z]{3,}", line) and spotlight_value_match(answer, line):
            start = offset + len(raw) - len(raw.lstrip())
            result.append(
                {
                    "line_number": number,
                    "char_start": start,
                    "char_end": start + len(line),
                    "text": line,
                }
            )
        offset += len(raw)
    return result


def candidate_line_retention(answer, context, selected):
    candidates = candidate_lines(answer, context)
    survived = any(normalize(c["text"]) in normalize(selected) for c in candidates)
    return {
        "metric": "loong_exact_candidate_line_survival_v2",
        "fraction_retained": float(survived),
        "all_retained": survived,
        "n_candidates": len(candidates),
        "annotation_status": "unreviewed_candidates",
        "evaluable": bool(candidates),
    }

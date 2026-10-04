"""Boundary tests for the LOONG value/entity matcher."""

from __future__ import annotations

from cora_bench.evaluation.loong_value_match import spotlight_value_match as M

# (gold, answer, expected)
CASES = [
    # magnitude must match exactly, with no 1-unit slack
    ("$101", "$100", False),
    ("1.03", "0.03", False),
    ("$100", "$99", False),
    # sign is lenient because financial text writes negatives as (x), -x, or x
    ("-$0.04", "$0.04", True),
    ("$0.04", "-$0.04", True),
    ("$(0.91)", "0.91", True),
    # correct positives
    ("$101", "the value is $101", True),
    ("-$0.04", "we get -0.04 per share", True),
    ("$(0.91)", "EPS was $(0.91)", True),  # parentheses = negative
    ("$(0.91)", "-0.91", True),
    ("2,647,583", "2647583", True),  # comma-insensitive
    ("12.5%", "12.5", True),  # percent
    # scale handling: gold states 'in thousands' -> raw OR scaled both count
    ("$10,135 in thousands", "10,135", True),
    ("$10,135 in thousands", "$10,135,000", True),
    ("$9,932 in thousands", "9,932,000", True),
    ("$9,932 in thousands", "the total was 9,932 (in thousands)", True),
    # entity: whole-word tokens, not substring-anywhere
    ("General Enterprise Ventures, Inc.", "The answer is General Enterprise Ventures.", True),
    ("Dominari Holdings Inc.", "Dominari Holdings", True),
    ("Apple Inc.", "pineapple orchard", False),  # 'apple' must not match inside 'pineapple'
]


def test_cases():
    fails = []
    for gold, ans, exp in CASES:
        got = M(gold, ans)
        if got != exp:
            fails.append((gold, ans, exp, got))
    if fails:
        for gold, ans, exp, got in fails:
            print(f"FAIL gold={gold!r} ans={ans!r} expected={exp} got={got}")
    assert not fails, f"{len(fails)} value-match cases failed"


if __name__ == "__main__":
    test_cases()
    print(f"PASS test_value_match ({len(CASES)} cases)")

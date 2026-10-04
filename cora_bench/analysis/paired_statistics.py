"""Paired cluster bootstrap over per-example evidence reports; failures count as zero.

NoLiMa resamples whole needle families, so related hops, placements, and haystacks stay together.
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cora_bench.utils.io import write_json


def cluster_id(row):
    if row["dataset"] == "nolima":
        return str(row["strata"]["needle_id"])
    return row["example_id"]


def paired_interval(left, right, *, field="fraction_retained", seed=0, repetitions=2000):
    paired_ids = sorted(left.keys() & right.keys())
    common = [
        eid
        for eid in paired_ids
        if left[eid].get("evidence_evaluable", True) and right[eid].get("evidence_evaluable", True)
    ]
    if not common:
        raise ValueError("No paired examples")
    groups = defaultdict(list)
    complete = 0
    for eid in common:
        a, b = left[eid], right[eid]
        if cluster_id(a) != cluster_id(b):
            raise ValueError("Pair has inconsistent cluster identity")
        good_a = not a.get("error") and a.get("evidence") is not None
        good_b = not b.get("error") and b.get("evidence") is not None
        complete += good_a and good_b
        av = float(a["evidence"][field]) if good_a else 0.0
        bv = float(b["evidence"][field]) if good_b else 0.0
        groups[cluster_id(a)].append(av - bv)
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    if len(sums) < 2:
        raise ValueError("At least two independent clusters required")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(sums), len(sums))
        estimates.append(sums[sample].sum() / sizes[sample].sum())
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {
        "difference_left_minus_right": float(sums.sum() / sizes.sum()),
        "ci95": [float(lo), float(hi)],
        "n_paired": len(common),
        "n_complete_pairs": int(complete),
        "n_clusters": len(groups),
        "n_unpaired_left": len(left) - len(paired_ids),
        "n_unpaired_right": len(right) - len(paired_ids),
        "n_missing_annotations": len(paired_ids) - len(common),
        "failure_policy": "zero",
        "bootstrap_unit": "needle_id"
        if next(iter(left.values()))["dataset"] == "nolima"
        else "example",
        "seed": seed,
        "bootstrap_repetitions": repetitions,
        "metric": field,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repetitions", type=int, default=2000)
    args = ap.parse_args()
    groups = defaultdict(dict)
    for line in Path(args.inp).open():
        r = json.loads(line)
        if r["answerable"]:
            groups[(r["method"], r["budget"])][r["example_id"]] = r
    result = []
    for a, b in itertools.combinations(groups, 2):
        if a[1] != b[1]:
            continue
        result.append(
            {
                "left": a[0],
                "right": b[0],
                "budget": a[1],
                **paired_interval(groups[a], groups[b], repetitions=args.repetitions),
            }
        )
    write_json(
        args.out,
        {
            "intervals": result,
            "interpretation": "Exploratory paired percentile intervals; no multiple-comparison significance claims.",
        },
    )


if __name__ == "__main__":
    main()

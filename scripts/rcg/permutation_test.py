# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Permutation tests over a per-seed evaluation CSV.

Makes no distributional assumption, which matters here: the baseline group is bimodal (working
runs plus collapsed ones), and a t-test on that assumes something the data plainly is not.

Two tests per comparison:

* **unpaired** -- shuffle group labels across the pooled values. Monte Carlo, since C(32,16) is
  ~6e8. The conservative, standard choice.
* **paired** -- seeds are matched across groups (same seed means the same initial policy weights
  and environment randomisation), so the sign-flip test applies and is more powerful. Enumerated
  exactly at 2^n, so the p-value is exact rather than sampled.

Also reports a rank-based effect size (probability a random draw from A exceeds one from B),
which is robust to the collapsed runs in a way a difference of means is not.

Usage:

.. code-block:: powershell

    isaaclab.bat -p scripts/rcg/permutation_test.py --csv eval_all_arms_p038.csv
    python scripts/rcg/permutation_test.py --csv eval_all_arms_p038.csv --metrics margin mean_max
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics


def _load(path: str) -> dict[str, dict[str, dict[str, float]]]:
    """Return {group: {seed: {metric: value}}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            vals = {}
            for k, v in row.items():
                if k in ("group", "run", "seed"):
                    continue
                try:
                    vals[k] = float(v)
                except (TypeError, ValueError):
                    vals[k] = float("nan")
            out.setdefault(row["group"], {})[row["seed"]] = vals
    return out


def _mc_unpaired(a: list[float], b: list[float], iters: int, rng) -> tuple[float, float]:
    """Two-sided Monte Carlo permutation on the difference of means."""
    obs = statistics.fmean(a) - statistics.fmean(b)
    pool = a + b
    na = len(a)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        diff = statistics.fmean(pool[:na]) - statistics.fmean(pool[na:])
        if abs(diff) >= abs(obs) - 1e-12:
            hits += 1
    # add-one correction: a Monte Carlo p-value should never be reported as exactly 0
    return obs, (hits + 1) / (iters + 1)


def _exact_paired(d: list[float]) -> tuple[float, float]:
    """Two-sided exact sign-flip permutation on paired differences."""
    obs = statistics.fmean(d)
    n = len(d)
    hits = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(statistics.fmean([s * x for s, x in zip(signs, d)])) >= abs(obs) - 1e-12:
            hits += 1
    return obs, hits / (2**n)


def _common_language(a: list[float], b: list[float]) -> float:
    """P(random draw from A > random draw from B), ties counted as half."""
    wins = sum((1.0 if x > y else 0.5 if x == y else 0.0) for x in a for y in b)
    return wins / (len(a) * len(b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation tests on a per-seed evaluation CSV.")
    parser.add_argument("--csv", type=str, required=True, help="CSV written by quick_drawer_eval.py.")
    parser.add_argument("--metrics", type=str, nargs="*", default=["margin", "mean_max", "success", "steps_to_p"])
    parser.add_argument("--iters", type=int, default=200_000, help="Monte Carlo samples for the unpaired test.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level flagged in the output.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the Monte Carlo test.")
    args = parser.parse_args()

    import random

    rng = random.Random(args.seed)

    data = _load(args.csv)
    groups = sorted(data)
    print(f"\nGroups: {', '.join(f'{g} (n={len(data[g])})' for g in groups)}")
    print(f"Unpaired: Monte Carlo, {args.iters:,} samples.  Paired: exact sign-flip.")

    for metric in args.metrics:
        print("\n" + "=" * 92)
        print(f"METRIC: {metric}")
        print("=" * 92)
        for g in groups:
            vals = [v[metric] for v in data[g].values() if v[metric] == v[metric]]
            n_nan = len(data[g]) - len(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            note = f"   ({n_nan} seed(s) with no value, excluded)" if n_nan else ""
            print(f"  {g:<22} mean {statistics.fmean(vals):9.4f}  sd {sd:8.4f}  median"
                  f" {statistics.median(vals):9.4f}{note}")

        print(f"\n  {'comparison':<28} {'diff':>10} {'p_unpaired':>12} {'p_paired':>10} {'P(A>B)':>9}")
        for ga, gb in itertools.combinations(groups, 2):
            a_by_seed, b_by_seed = data[ga], data[gb]
            a = [v[metric] for v in a_by_seed.values() if v[metric] == v[metric]]
            b = [v[metric] for v in b_by_seed.values() if v[metric] == v[metric]]
            if len(a) < 2 or len(b) < 2:
                print(f"  {ga+' vs '+gb:<28} insufficient data")
                continue

            diff, p_unpaired = _mc_unpaired(list(a), list(b), args.iters, rng)

            shared = [s for s in a_by_seed if s in b_by_seed]
            paired = [
                a_by_seed[s][metric] - b_by_seed[s][metric]
                for s in shared
                if a_by_seed[s][metric] == a_by_seed[s][metric] and b_by_seed[s][metric] == b_by_seed[s][metric]
            ]
            if 2 <= len(paired) <= 20:
                _, p_paired = _exact_paired(paired)
                p_paired_s = f"{p_paired:.4f}"
            else:
                p_paired, p_paired_s = float("nan"), "n/a"

            cl = _common_language(a, b)
            stars = ""
            if p_unpaired < args.alpha:
                stars += " *unpaired"
            if p_paired == p_paired and p_paired < args.alpha:
                stars += " *paired"
            print(f"  {ga+' vs '+gb:<28} {diff:>+10.4f} {p_unpaired:>12.4f} {p_paired_s:>10} {cl:>9.3f}{stars}")

    print(f"\n* = p < {args.alpha}.  P(A>B) is the probability a random seed from A beats one from B;")
    print("  0.5 means no separation, and it is unaffected by how extreme the collapsed runs are.")


if __name__ == "__main__":
    main()

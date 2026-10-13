#!/usr/bin/env python3
"""Audit saved study artifacts without training or importing run.py.

Run from any directory. The script reads artifacts and prints a JSON receipt.
It independently recomputes every non-random selected mask on every pair.
It checks other candidate rows and random selections against the saved tables.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
LOCK = "ff9d78c15aa6e97b4819e18f4253b60fd165b50c"
MAX_ERROR = 0.0
CHECKED_SCALARS = 0


def table(relative):
    with (STUDY / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def equal(actual, expected):
    global MAX_ERROR, CHECKED_SCALARS
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    MAX_ERROR = max(MAX_ERROR, error)
    CHECKED_SCALARS += a.size
    assert error < 5e-12, (error, actual, expected)


def encode(states):
    return np.concatenate([np.eye(size)[states[:, column]]
                           for column, size in enumerate((4, 4, 2, 6))], axis=1)


def downstream(hidden, w):
    middle = np.tanh(hidden @ w["W2"] + w["b2"])
    logits = middle @ w["W3"] + w["b3"]
    ex = np.exp(logits - np.max(logits, axis=1)[:, None])
    return ex / np.sum(ex, axis=1)[:, None]


def distance(a, b):
    return np.sum(np.abs(a - b), axis=1) / 2


def decode_mask(text):
    return tuple(int(number) for number in text.split())


def audit():
    manifest = json.loads((STUDY / "results/manifest.json").read_text())
    for path, expected in manifest.items():
        assert hashlib.sha256((STUDY / path).read_bytes()).hexdigest() == expected, path
    lock = json.loads((STUDY / "results/lock.json").read_text())
    assert lock["commit"] == LOCK
    for path, expected in lock["sha256"].items():
        current = (REPO / path).read_bytes()
        assert hashlib.sha256(current).hexdigest() == expected
        saved = subprocess.check_output(["git", "show", LOCK + ":" + path], cwd=REPO)
        assert saved == current, path

    state_rows = table("data/inputs.csv")
    states = np.asarray([[int(row[key]) for key in ("a", "b", "s", "n")]
                         for row in state_rows])
    expected_states = np.asarray(list(itertools.product(range(4), range(4), range(2), range(6))))
    assert np.array_equal(states, expected_states)
    assert [int(row["id"]) for row in state_rows] == list(range(192))
    labels = np.asarray([row[0] if row[2] == 0 else row[1] for row in states])
    assert np.array_equal(labels, [int(row["label"]) for row in state_rows])
    x = encode(states)

    pair_rows = table("data/pairs.csv")
    r = np.asarray([int(row["receiver"]) for row in pair_rows])
    d = np.asarray([int(row["donor"]) for row in pair_rows])
    pair_ids = r * 192 + d
    expected_pairs = {(i, j) for i in range(192) for j in range(192)
                      if states[i, 0] != states[j, 0]}
    assert len(pair_rows) == len(set(pair_ids)) == len(expected_pairs) == 27648
    assert set(zip(r, d)) == expected_pairs
    assert np.array_equal(pair_ids, [int(row["pair_id"]) for row in pair_rows])
    active = states[r, 2] == 0
    sm, bm = states[r, 2] == states[d, 2], states[r, 1] == states[d, 1]
    strata = states[r, 2] * 4 + sm.astype(int) * 2 + bm.astype(int)
    matched = sm & bm & (states[r, 3] == states[d, 3])
    test = np.asarray([row["split"] == "test" for row in pair_rows])
    assert set(row["split"] for row in pair_rows) == {"test", "discovery"}
    assert np.array_equal(strata, [int(row["stratum"]) for row in pair_rows])
    assert np.array_equal(matched, [row["matched"] == "True" for row in pair_rows])
    assert len(set(pair_ids[test]) & set(pair_ids[~test])) == 0
    stratum_counts = []
    for group in range(8):
        indices = np.where(strata == group)[0]
        ordered = sorted(indices, key=lambda i: hashlib.sha256(
            f"inactive-variable-audit-v1:{r[i]}:{d[i]}".encode()).digest())
        assert not test[ordered[:len(ordered)//2]].any()
        assert test[ordered[len(ordered)//2:]].all()
        stratum_counts.append(int(len(indices)))
    assert test.sum() == 13824 and matched.sum() == 576
    cf_states = states[r].copy()
    cf_states[:, 0] = states[d, 0]
    state_lookup = {tuple(row): index for index, row in enumerate(states)}
    cf = np.asarray([state_lookup[tuple(row)] for row in cf_states])
    assert np.array_equal(cf, [int(row["counterfactual"]) for row in pair_rows])

    masks = [mask for size in (1, 2, 3) for mask in itertools.combinations(range(12), size)]
    selected = table("results/selected-masks.csv")
    assert len(selected) == 928
    predictions = table("results/intact-predictions.csv")
    outcomes = table("results/seed-outcomes.csv")
    same_controls = table("results/same-a-controls.csv")
    baseline_rows = table("results/empty-baselines.csv")
    groups = {(regime, split): (np.ones(len(r), dtype=bool) if regime == "unrestricted" else matched)
              & (test if split == "test" else ~test)
              for regime in ("unrestricted", "matched") for split in ("discovery", "test")}
    same_pairs = np.asarray([(i, j) for i in range(192) for j in range(192)
                             if states[i, 0] == states[j, 0]])
    assert len(same_pairs) == 9216
    differences = []
    nonrandom_count = 0
    raw_count = 0
    for seed in range(8):
        raw = table(f"results/candidate-scores-seed-{seed}.csv")
        assert len(raw) == 1192
        raw_count += len(raw)
        by_key = {(row["regime"], row["split"], int(row["mask_id"])): row for row in raw}
        assert len(by_key) == len(raw)
        for row in raw:
            mid = int(row["mask_id"])
            assert decode_mask(row["mask"]) == masks[mid]
            assert int(row["k"]) == len(masks[mid])
            assert int(row["seed"]) == seed
            keep = groups[(row["regime"], row["split"])]
            assert int(row["n"]) == keep.sum()
            equal(float(row["worst_tv"]), max(float(row["active_tv"]), float(row["inactive_tv"])))
            fraction = (keep & active).sum() / keep.sum()
            equal(float(row["tv"]), fraction * float(row["active_tv"])
                  + (1-fraction) * float(row["inactive_tv"]))

        chosen = [row for row in selected if int(row["seed"]) == seed]
        assert len(chosen) == 116
        assert len({(row["regime"], row["k"], row["method"]) for row in chosen}) == 116
        for row in chosen:
            size, mid = int(row["k"]), int(row["mask_id"])
            regime, method = row["regime"], row["method"]
            pool = [z for z in raw if z["regime"] == regime and z["split"] == "discovery"
                    and int(z["k"]) == size]
            pool.sort(key=lambda z: decode_mask(z["mask"]))
            if method in ("repair_only", "balanced"):
                key = "active_tv" if method == "repair_only" else "worst_tv"
                minimum = min(float(z[key]) for z in pool)
                expected = next(z for z in pool if float(z[key]) <= minimum + 1e-12)
                assert int(expected["mask_id"]) == mid
                nonrandom_count += 1
            else:
                assert method.startswith("random_")
                indices = sorted(np.random.default_rng(9000+size).choice(
                    len(pool), size=min(20, len(pool)), replace=False))
                assert mid == int(pool[indices[int(method.split("_")[1])]]["mask_id"])
            assert decode_mask(row["mask"]) == masks[mid]
            for split in ("discovery", "test"):
                reference = by_key[(regime, split, mid)]
                for key in row:
                    if key.startswith(split + "_"):
                        equal(float(row[key]), float(reference[key[len(split)+1:]]))

        with np.load(STUDY / f"models/seed-{seed}.npz", allow_pickle=False) as archive:
            w = {key: archive[key] for key in archive.files}
        for key, shape in {"W1": (16,12), "b1": (12,), "W2": (12,24),
                           "b2": (24,), "W3": (24,4), "b3": (4,)}.items():
            assert w[key].shape == shape and w[key].dtype == np.float64
        h = np.tanh(x @ w["W1"] + w["b1"])
        p = downstream(h, w)
        saved = sorted([z for z in predictions if int(z["seed"]) == seed], key=lambda z: int(z["input_id"]))
        assert len(saved) == 192
        equal(p, [[float(z[f"p{k}"]) for k in range(4)] for z in saved])
        accuracy = float(np.mean(np.argmax(p, axis=1) == labels))
        eligible = accuracy >= .99
        assert all((z["eligible"] == "True") == eligible for z in raw + chosen)
        equal(downstream(np.tanh(encode(cf_states) @ w["W1"] + w["b1"]), w), p[cf])
        equal(downstream(h[r], w), p[r])
        equal(downstream(h[d], w), p[d])
        target = p[cf]
        intact_change = distance(target, p[r])

        def metrics(pred, keep):
            err, change = distance(pred, target), distance(pred, p[r])
            result = {"n": int(keep.sum()), "tv": err[keep].mean(),
                      "active_tv": err[keep & active].mean(),
                      "inactive_tv": err[keep & ~active].mean(),
                      "semantic_accuracy": np.mean(pred.argmax(1)[keep] == labels[cf][keep]),
                      "donor_copy_rate": np.mean(pred.argmax(1)[keep] == labels[d][keep]),
                      "inactive_output_change_tv": change[keep & ~active].mean(),
                      "intact_inactive_change_tv": intact_change[keep & ~active].mean()}
            result["worst_tv"] = max(result["active_tv"], result["inactive_tv"])
            for group in range(8):
                loc = keep & (strata == group)
                if loc.any():
                    result[f"stratum_{group}_tv"] = err[loc].mean()
            return result

        audited_ids = sorted({int(z["mask_id"]) for z in chosen if not z["method"].startswith("random_")})
        for mid in audited_ids:
            mixed = h[r].copy()
            mixed[:, masks[mid]] = h[d][:, masks[mid]]
            patched = downstream(mixed, w)
            for (regime, split), keep in groups.items():
                reference = by_key[(regime, split, mid)]
                for key, value in metrics(patched, keep).items():
                    equal(value, float(reference[key]))
            sr, sd = same_pairs.T
            mixed_same = h[sr].copy()
            mixed_same[:, masks[mid]] = h[sd][:, masks[mid]]
            same_p = downstream(mixed_same, w)
            same_err = distance(same_p, p[sr])
            same_active = states[sr, 2] == 0
            reference = next(z for z in same_controls if int(z["seed"]) == seed and int(z["mask_id"]) == mid)
            assert int(reference["n"]) == 9216
            for key, value in {"tv": same_err.mean(), "active_tv": same_err[same_active].mean(),
                               "inactive_tv": same_err[~same_active].mean(),
                               "changed_answer_rate": np.mean(same_p.argmax(1) != p[sr].argmax(1))}.items():
                equal(value, float(reference[key]))
        for (regime, split), keep in groups.items():
            reference = next(z for z in baseline_rows if int(z["seed"]) == seed
                             and z["regime"] == regime and z["split"] == split)
            for key, value in metrics(p[r], keep).items():
                equal(value, float(reference[key]))

        repair = next(z for z in chosen if z["regime"] == "unrestricted" and z["k"] == "3" and z["method"] == "repair_only")
        balanced = next(z for z in chosen if z["regime"] == "unrestricted" and z["k"] == "3" and z["method"] == "balanced")
        outcome = next(z for z in outcomes if int(z["seed"]) == seed)
        equal(accuracy, float(outcome["accuracy"]))
        for prefix, row in (("repair", repair), ("balanced", balanced)):
            assert outcome[prefix + "_mask"] == row["mask"]
            for key in ("active_tv", "inactive_tv", "worst_tv"):
                equal(float(outcome[prefix + "_" + key]), float(row["test_" + key]))
        difference = float(repair["test_worst_tv"]) - float(balanced["test_worst_tv"])
        equal(difference, float(outcome["improvement"]))
        if eligible:
            differences.append(difference)

    summary = json.loads((STUDY / "results/summary.json").read_text())
    assert summary["eligible_seed_count"] == len(differences)
    assert summary["gate_passed"] == (len(differences) >= 6)
    primary = summary["primary"]
    for key, value in {"n": len(differences), "mean": np.mean(differences),
                       "median": np.median(differences), "minimum": min(differences),
                       "maximum": max(differences)}.items():
        equal(float(primary[key]), value)
    draws = np.random.default_rng(20261013).choice(differences, size=(10000, len(differences)), replace=True)
    equal(primary["ci95"], np.percentile(np.mean(draws, axis=1), [2.5,97.5]))
    return {"passed": True, "lock_commit": LOCK, "manifest_files": len(manifest),
            "states": 192, "pairs": len(pair_rows), "test_pairs": int(test.sum()),
            "matched_test_pairs": int((test & matched).sum()), "stratum_counts": stratum_counts,
            "seeds": 8, "candidate_rows_checked": raw_count,
            "selected_rows_checked": len(selected), "nonrandom_selection_configurations": nonrandom_count,
            "forward_check_scope": "Every selected non-random mask, every primary and same-a pair, all eight seeds",
            "checked_numeric_values": int(CHECKED_SCALARS), "maximum_absolute_difference": MAX_ERROR,
            "primary_seed_differences": differences,
            "limits": "No independent training; no exhaustive forward recalculation of every unselected or random-only mask."}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))

#!/usr/bin/env python3
"""Finite basis-transport audit. --check does not train or inspect trained models."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
SEEDS = tuple(range(8))
STEPS = 4000
LR = 0.02
TIE_TOL = 1e-12
CHECK_TOL = 1e-10
P = np.diag([1., 1., 1., 1., 0., 0., 0., 0.])
MASKS = tuple(itertools.combinations(range(8), 4))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No rows: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inputs():
    states = np.array(list(itertools.product(range(5), repeat=2)), dtype=np.int64)
    x = np.concatenate([np.eye(5)[states[:, 0]], np.eye(5)[states[:, 1]]], axis=1)
    y = 5*((states[:, 0]+states[:, 1]) % 5) + states[:, 1]
    return states, x, y


def pair_bank(states):
    r, d = np.indices((25, 25))
    r, d = r.ravel(), d.ravel()
    keep = states[r, 0] != states[d, 0]
    r, d = r[keep], d[keep]
    hashes = [sha(f"basis-audit-v1|{a}|{b}".encode("utf-8")) for a, b in zip(r, d)]
    order = sorted(range(len(r)), key=lambda i: (hashes[i], int(r[i]*25+d[i])))
    test = np.zeros(len(r), dtype=bool)
    test[order[250:]] = True
    return dict(r=r, d=d, cf=5*states[d, 0]+states[r, 1], test=test,
                b_match=states[r, 1] == states[d, 1], hash=hashes)


def bases():
    result = [dict(basis_id=0, name="identity", kind="identity", rotation_seed=-1,
                   condition=1., T=np.eye(8))]
    rng = np.random.default_rng(99)
    perm = rng.permutation(8)
    signs = rng.choice([-1., 1.], size=8)
    signed = np.zeros((8, 8))
    signed[np.arange(8), perm] = signs
    result.append(dict(basis_id=1, name="signed_permutation", kind="permutation",
                       rotation_seed=99, condition=1., T=signed))
    rotations = []
    for seed in range(100, 108):
        q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(8, 8)))
        q *= np.where(np.diag(r) < 0, -1., 1.)[None, :]
        rotations.append((seed, q))
    for condition in (1., 4., 16.):
        for seed, q in rotations:
            scale = np.geomspace(condition**-.5, condition**.5, 8)
            transform = q @ np.diag(scale)
            result.append(dict(basis_id=len(result),
                               name=f"rotation_{seed}" if condition == 1 else f"scaled_{int(condition)}_{seed}",
                               kind="rotation" if condition == 1 else f"condition_{int(condition)}",
                               rotation_seed=seed, condition=condition, T=transform))
    return result


def initialize(seed):
    rng = np.random.default_rng(seed)
    p = {}
    for weight, bias, n_in, n_out in (("Ea", "ba", 5, 4), ("Eb", "bb", 5, 4),
                                     ("W1", "b1", 8, 32), ("W2", "b2", 32, 25)):
        p[weight] = rng.normal(0., np.sqrt(2./(n_in+n_out)), size=(n_in, n_out))
        p[bias] = np.zeros(n_out)
    return p


def encode(x, p):
    return np.concatenate([x[:, :5] @ p["Ea"] + p["ba"],
                           x[:, 5:] @ p["Eb"] + p["bb"]], axis=1)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex/ex.sum(axis=1, keepdims=True)


def decode(z, p, first_weight=None):
    w = p["W1"] if first_weight is None else first_weight
    h = np.tanh(z @ w + p["b1"])
    return softmax(h @ p["W2"] + p["b2"])


def forward(x, p):
    z = encode(x, p)
    h = np.tanh(z @ p["W1"] + p["b1"])
    logits = h @ p["W2"] + p["b2"]
    return softmax(logits), z, h, logits


def loss_grad(x, y, p):
    prob, z, h, logits = forward(x, p)
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    dz = prob.copy()
    dz[np.arange(len(y)), y] -= 1
    dz /= len(y)
    g = dict(W2=h.T @ dz, b2=dz.sum(axis=0))
    dh = (dz @ p["W2"].T)*(1-h*h)
    g.update(W1=z.T @ dh, b1=dh.sum(axis=0))
    de = dh @ p["W1"].T
    g.update(Ea=x[:, :5].T @ de[:, :4], ba=de[:, :4].sum(axis=0),
             Eb=x[:, 5:].T @ de[:, 4:], bb=de[:, 4:].sum(axis=0))
    return float(loss), g


def train(seed, x, y):
    p = initialize(seed)
    first = {k: np.zeros_like(v) for k, v in p.items()}
    second = {k: np.zeros_like(v) for k, v in p.items()}
    trace = []
    for step in range(1, STEPS+1):
        loss, grad = loss_grad(x, y, p)
        for name in p:
            first[name] = .9*first[name] + .1*grad[name]
            second[name] = .999*second[name] + .001*grad[name]*grad[name]
            p[name] -= LR*(first[name]/(1-.9**step))/(np.sqrt(second[name]/(1-.999**step))+1e-8)
        if step == 1 or step % 100 == 0:
            trace.append(dict(seed=seed, step=step, loss_before_update=loss,
                              accuracy_after_update=float(np.mean(forward(x,p)[0].argmax(1)==y))))
    return p, trace


def tv(a, b):
    return .5*np.abs(a-b).sum(axis=1)


def projection(mask):
    return np.diag(np.isin(np.arange(8), mask).astype(float))


def patch(z, p, pair, operator, first_weight=None):
    r, d = pair["r"], pair["d"]
    return decode(z[r]+(z[d]-z[r]) @ operator, p, first_weight)


def transported(p, z, transform):
    z_new = z @ transform
    first_weight = np.linalg.solve(transform, p["W1"])
    operator = np.linalg.solve(transform, P @ transform)
    a = transform[:4, :]
    orthogonal_image = a.T @ np.linalg.solve(a @ a.T, a)
    return z_new, first_weight, operator, orthogonal_image


def metrics(prediction, target, labels, pair, keep):
    errors = tv(prediction, target)
    result = dict(n=int(keep.sum()), tv=float(errors[keep].mean()),
                  semantic_accuracy=float(np.mean(prediction[keep].argmax(1)==labels[keep])))
    for matched, label in ((True, "b_match"), (False, "b_mismatch")):
        subset = keep & (pair["b_match"] == matched)
        result[label+"_n"] = int(subset.sum())
        result[label+"_tv"] = float(errors[subset].mean()) if subset.any() else None
    return result


def controls(p, x, pair, all_bases, seed):
    prob, z, _, _ = forward(x, p)
    r, d = pair["r"], pair["d"]
    original = patch(z, p, pair, P)
    z_patch = z[r]+(z[d]-z[r]) @ P
    branch_interface_error = float(np.max(np.abs(z_patch-z[pair["cf"]])))
    branch_probability_error = float(np.max(np.abs(original-prob[pair["cf"]])))
    assert max(branch_interface_error, branch_probability_error) < CHECK_TOL
    rows = []
    for basis in all_bases:
        t = basis["T"]
        zn, w, op, e = transported(p, z, t)
        out = decode(zn, p, w)
        transported_out = patch(zn, p, pair, op, w)
        # Separate dense-encoder path; a dense transform need not preserve branch sparsity.
        encoder = np.zeros((10, 8))
        encoder[:5, :4] = p["Ea"]
        encoder[5:, 4:] = p["Eb"]
        bias = np.concatenate([p["ba"], p["bb"]])
        folded_z = x @ (encoder @ t) + bias @ t
        folded_out = decode(folded_z, p, w)
        actual_condition = float(np.linalg.cond(t))
        row = dict(seed=seed, basis_id=basis["basis_id"], basis=basis["name"], kind=basis["kind"],
                   declared_condition=basis["condition"], actual_condition=actual_condition,
                   branch_interface_residual=branch_interface_error,
                   branch_probability_residual=branch_probability_error,
                   intact_residual=float(np.max(np.abs(out-prob))),
                   folded_encoder_residual=float(np.max(np.abs(folded_out-prob))),
                   transport_residual=float(np.max(np.abs(transported_out-original))),
                   empty_residual=float(np.max(np.abs(patch(zn,p,pair,np.zeros((8,8)),w)-prob[r]))),
                   full_residual=float(np.max(np.abs(patch(zn,p,pair,np.eye(8),w)-prob[d]))),
                   idempotence_residual=float(np.max(np.abs(op @ op-op))),
                   image_idempotence_residual=float(np.max(np.abs(e @ e-e))),
                   image_symmetry_residual=float(np.max(np.abs(e-e.T))),
                   transport_symmetry_residual=float(np.max(np.abs(op-op.T))),
                   same_image_residual=float(np.max(np.abs(op @ e-op))),
                   operator_difference_fro=float(np.linalg.norm(e-op)),
                   orthogonality_residual=float(np.max(np.abs(t.T @ t-np.eye(8)))),
                   b_match_image_counterfactual_residual=float(np.max(np.abs(
                       patch(zn,p,pair,e,w)[pair["b_match"]]-prob[pair["cf"]][pair["b_match"]]))),
                   permutation_family_residual=0., scaling_family_residual=0.)
        required = [key for key in row if key.endswith("residual") and key not in (
            "transport_symmetry_residual", "orthogonality_residual")]
        assert max(row[key] for key in required) < CHECK_TOL, row
        assert abs(actual_condition-basis["condition"]) < CHECK_TOL
        if basis["condition"] == 1:
            assert row["orthogonality_residual"] < CHECK_TOL
            assert row["transport_symmetry_residual"] < CHECK_TOL
            assert row["operator_difference_fro"] < CHECK_TOL
        if basis["kind"] == "permutation":
            largest = 0.
            for mask in MASKS:
                current_operator = projection(mask)
                native_operator = t @ current_operator @ t.T
                native_mask = tuple(np.flatnonzero(np.diag(native_operator) > .5))
                assert len(native_mask) == 4
                assert np.max(np.abs(native_operator-projection(native_mask))) < CHECK_TOL
                lhs = patch(zn, p, pair, current_operator, w)
                rhs = patch(z, p, pair, projection(native_mask))
                largest = max(largest, float(np.max(np.abs(lhs-rhs))))
            row["permutation_family_residual"] = largest
            assert largest < CHECK_TOL
        if basis["condition"] > 1:
            scale = np.geomspace(basis["condition"]**-.5, basis["condition"]**.5, 8)
            q = t @ np.diag(1./scale)
            qz = z @ q
            qw = np.linalg.solve(q, p["W1"])
            largest = 0.
            for mask in MASKS:
                coordinate_operator = projection(mask)
                lhs = patch(zn, p, pair, coordinate_operator, w)
                rhs = patch(qz, p, pair, coordinate_operator, qw)
                largest = max(largest, float(np.max(np.abs(lhs-rhs))))
            row["scaling_family_residual"] = largest
            assert largest < CHECK_TOL
        rows.append(row)
    return rows


def correctness_checks():
    states, x, y = inputs()
    pair = pair_bank(states)
    all_bases = bases()
    assert x.shape == (25, 10) and len(set(y)) == 25
    assert len(pair["r"]) == 500 and int(pair["test"].sum()) == 250
    assert int(pair["b_match"].sum()) == 100
    ids = pair["r"]*25+pair["d"]
    assert len(set(ids)) == 500
    assert set(ids[pair["test"]]).isdisjoint(ids[~pair["test"]])
    assert len(MASKS) == 70 and len(all_bases) == 26
    assert np.all(states[pair["cf"],0] == states[pair["d"],0])
    assert np.all(states[pair["cf"],1] == states[pair["r"],1])
    p = initialize(314)
    batch = np.array([0, 3, 8, 11, 14, 20, 24])
    _, gradient = loss_grad(x[batch], y[batch], p)
    locations = [(name, ix) for name, value in p.items() for ix in np.ndindex(value.shape)]
    samples = np.random.default_rng(2718).choice(len(locations), size=40, replace=False)
    largest = 0.
    for sample in samples:
        name, ix = locations[sample]
        before = p[name][ix]
        p[name][ix] = before+1e-5
        plus = loss_grad(x[batch], y[batch], p)[0]
        p[name][ix] = before-1e-5
        minus = loss_grad(x[batch], y[batch], p)[0]
        p[name][ix] = before
        estimate = (plus-minus)/2e-5
        largest = max(largest, float(abs(estimate-gradient[name][ix])))
    assert largest < 1e-6, largest
    checked = controls(p, x, pair, all_bases, seed=314)
    return dict(passed=True, trained=False, gradient_coordinates=40,
                gradient_max_absolute_error=largest, task_states=25, pairs=500,
                discovery_pairs=250, test_pairs=250, coordinate_masks=70, bases=26,
                b_match_discovery=int(np.sum(pair["b_match"] & ~pair["test"])),
                b_match_test=int(np.sum(pair["b_match"] & pair["test"])),
                untrained_max_transport_residual=max(z["transport_residual"] for z in checked),
                untrained_max_permutation_family_residual=max(z["permutation_family_residual"] for z in checked),
                untrained_max_scaling_family_residual=max(z["scaling_family_residual"] for z in checked))


def evaluate(seed, p, x, y, pair, all_bases, results):
    prob, z, _, _ = forward(x, p)
    eligible = bool(np.mean(prob.argmax(1)==y) >= .99)
    target = prob[pair["cf"]]
    labels = y[pair["cf"]]
    candidates, selected, operator_rows = [], [], []
    for basis in all_bases:
        zn, w, op, e = transported(p, z, basis["T"])
        common = dict(seed=seed, eligible=eligible, basis_id=basis["basis_id"],
                      basis=basis["name"], kind=basis["kind"],
                      rotation_seed=basis["rotation_seed"], condition=basis["condition"])
        local = []
        for mask_id, mask in enumerate(MASKS):
            prediction = patch(zn, p, pair, projection(mask), w)
            for split in ("discovery", "test"):
                keep = pair["test"] == (split == "test")
                row = dict(**common, mask_id=mask_id, mask=" ".join(map(str,mask)), split=split,
                           **metrics(prediction,target,labels,pair,keep))
                candidates.append(row)
                local.append(row)
        discovery = [row for row in local if row["split"] == "discovery"]
        minimum = min(row["tv"] for row in discovery)
        winner = next(row for row in discovery if row["tv"] <= minimum+TIE_TOL)
        test = next(row for row in local if row["split"] == "test" and row["mask_id"] == winner["mask_id"])
        selected.append(dict(**common, mask_id=winner["mask_id"], mask=winner["mask"],
                             discovery_minimum_tv=minimum, discovery_tv=winner["tv"],
                             **{"test_"+key: test[key] for key in (
                                 "n", "tv", "semantic_accuracy", "b_match_n", "b_match_tv", "b_mismatch_n", "b_mismatch_tv")}))
        # E is reported for every basis; in orthogonal bases it is also a positive control.
        for method, operator in (("transported", op), ("orthogonal_image", e),
                                 ("empty", np.zeros((8, 8))), ("full", np.eye(8))):
            prediction = patch(zn, p, pair, operator, w)
            for split in ("discovery", "test"):
                keep = pair["test"] == (split == "test")
                operator_rows.append(dict(**common, method=method, split=split,
                                          **metrics(prediction,target,labels,pair,keep)))
    write_csv(results/f"candidate-scores-seed-{seed}.csv", candidates)
    return selected, operator_rows


def descriptives(values):
    values = np.array(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()), minimum=float(values.min()),
                maximum=float(values.max()))


def make_summary(selected, operators, seed_rows, control_rows):
    kinds = ("identity", "permutation", "rotation", "condition_4", "condition_16")
    per_seed = []
    for seed in SEEDS:
        original = next(row for row in selected if row["seed"] == seed and row["kind"] == "identity")
        for kind in kinds:
            current = [row for row in selected if row["seed"] == seed and row["kind"] == kind]
            record = dict(seed=seed, eligible=original["eligible"], kind=kind,
                          basis_count=len(current),
                          coordinate_tv_mean=float(np.mean([z["test_tv"] for z in current])),
                          coordinate_tv_min=float(np.min([z["test_tv"] for z in current])),
                          coordinate_tv_max=float(np.max([z["test_tv"] for z in current])),
                          coordinate_accuracy_mean=float(np.mean([z["test_semantic_accuracy"] for z in current])))
            record["coordinate_tv_change_from_identity"] = record["coordinate_tv_mean"]-original["test_tv"]
            for method in ("transported", "orthogonal_image"):
                values = [row for row in operators if row["seed"] == seed and row["kind"] == kind
                          and row["method"] == method and row["split"] == "test"]
                record[method+"_tv_mean"] = float(np.mean([z["tv"] for z in values]))
                record[method+"_accuracy_mean"] = float(np.mean([z["semantic_accuracy"] for z in values]))
            per_seed.append(record)
    aggregate = {}
    for kind in kinds:
        aggregate[kind] = {metric: descriptives([row[metric] for row in per_seed if row["kind"] == kind])
                           for metric in ("coordinate_tv_mean", "coordinate_accuracy_mean",
                                          "coordinate_tv_change_from_identity", "transported_tv_mean",
                                          "orthogonal_image_tv_mean", "orthogonal_image_accuracy_mean")}
    return dict(question="How does a coordinate change alter the candidate family and the intervention operator?",
                aggregation="All eight seeds; each seed has equal weight. Rotations are repeated conditions, not independent trained models.",
                eligible_seed_count=sum(row["eligible"] for row in seed_rows),
                all_seed_count=len(seed_rows), across_seed=aggregate, per_seed=per_seed,
                max_intact_residual=max(row["intact_residual"] for row in control_rows),
                max_folded_encoder_residual=max(row["folded_encoder_residual"] for row in control_rows),
                max_transport_residual=max(row["transport_residual"] for row in control_rows),
                max_permutation_family_residual=max(row["permutation_family_residual"] for row in control_rows),
                max_scaling_family_residual=max(row["scaling_family_residual"] for row in control_rows)), per_seed


def verify_lock(commit):
    if not commit:
        raise ValueError("--run requires --lock-commit. Publish the plan and runner before training.")
    resolved = subprocess.check_output(["git", "rev-parse", commit+"^{commit}"],cwd=REPO,text=True).strip()
    hashes = {}
    for file in (STUDY/"analysis-plan.md", Path(__file__).resolve()):
        relative = file.relative_to(REPO).as_posix()
        locked = subprocess.check_output(["git", "show", resolved+":"+relative],cwd=REPO)
        if file.read_bytes() != locked:
            raise ValueError("File differs from lock: "+relative)
        hashes[relative] = sha(locked)
    return dict(commit=resolved, sha256=hashes)


def run(commit):
    lock = verify_lock(commit)
    result = STUDY/"results"
    model_dir = STUDY/"models"
    for directory in (result, model_dir, STUDY/"data"):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Refuse to overwrite existing scientific outputs: "+str(directory))
    started = time.monotonic()
    preflight = correctness_checks()
    states, x, y = inputs()
    pair, all_bases = pair_bank(states), bases()
    result.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (STUDY/"data").mkdir(parents=True, exist_ok=True)
    write_json(result/"lock.json", lock)
    write_json(result/"preflight-controls.json", preflight)
    write_csv(STUDY/"data"/"inputs.csv", [dict(id=i,a=int(s[0]),b=int(s[1]),label=int(y[i])) for i,s in enumerate(states)])
    write_csv(STUDY/"data"/"pairs.csv", [dict(pair_id=int(r*25+d),receiver=int(r),donor=int(d),
              counterfactual=int(pair["cf"][i]),b_match=bool(pair["b_match"][i]),
              split="test" if pair["test"][i] else "discovery",hash=pair["hash"][i])
              for i,(r,d) in enumerate(zip(pair["r"],pair["d"]))])
    write_csv(STUDY/"data"/"bases.csv", [{key:value for key,value in basis.items() if key != "T"} for basis in all_bases])
    np.savez(STUDY/"data"/"transforms.npz", matrices=np.array([b["T"] for b in all_bases]),
             names=np.array([b["name"] for b in all_bases]))
    selected, operators, traces, seed_rows, control_rows, intact_rows = [], [], [], [], [], []
    for seed in SEEDS:
        seed_start = time.monotonic()
        p, trace = train(seed, x, y)
        np.savez(model_dir/f"seed-{seed}.npz", **p)
        traces.extend(trace)
        prob = forward(x,p)[0]
        accuracy = float(np.mean(prob.argmax(1)==y))
        checked = controls(p,x,pair,all_bases,seed)
        control_rows.extend(checked)
        chosen, operator_metrics = evaluate(seed,p,x,y,pair,all_bases,result)
        selected.extend(chosen)
        operators.extend(operator_metrics)
        seed_row = dict(seed=seed,eligible=accuracy>=.99,accuracy=accuracy,
                        final_loss=loss_grad(x,y,p)[0],seconds=time.monotonic()-seed_start)
        seed_rows.append(seed_row)
        for i, row in enumerate(prob):
            intact_rows.append(dict(seed=seed,input_id=i,label=int(y[i]),prediction=int(row.argmax()),
                                    **{f"p{j}":float(value) for j,value in enumerate(row)}))
        write_csv(result/"seed-outcomes.csv",seed_rows)
        print(json.dumps(seed_row),flush=True)
    write_csv(result/"selected-masks.csv",selected)
    write_csv(result/"operator-metrics.csv",operators)
    write_csv(result/"controls.csv",control_rows)
    write_csv(result/"training-trace.csv",traces)
    write_csv(result/"intact-predictions.csv",intact_rows)
    summary, per_seed = make_summary(selected,operators,seed_rows,control_rows)
    write_csv(result/"per-seed-summary.csv",per_seed)
    summary.update(elapsed_seconds=time.monotonic()-started,lock=lock,
                   environment=dict(python=platform.python_version(),numpy=np.__version__,
                                    platform=platform.platform(),blas_threads=1))
    write_json(result/"summary.json",summary)
    assert verify_lock(commit) == lock
    files = sorted(list((STUDY/"data").glob("*"))+list(model_dir.glob("*"))+list(result.glob("*")))
    write_json(result/"manifest.json",{f.relative_to(STUDY).as_posix():sha(f.read_bytes())
                                       for f in files if f.name != "manifest.json"})
    print(json.dumps({key:summary[key] for key in ("eligible_seed_count","max_intact_residual",
                      "max_transport_residual","max_permutation_family_residual","elapsed_seconds")},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--run",action="store_true")
    parser.add_argument("--lock-commit")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(correctness_checks(),indent=2))
    else:
        run(args.lock_commit)


if __name__ == "__main__":
    main()

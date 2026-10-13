#!/usr/bin/env python3
"""Finite, CPU-only inactive-variable audit. --check never trains a model."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

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
SEEDS = list(range(8))
STEPS = 4000
LR = 0.02
TOL = 1e-12


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("Refuse to write an empty table: " + str(path))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_inputs():
    states = np.array(list(itertools.product(range(4), range(4), range(2), range(6))),
                      dtype=np.int64)
    x = np.concatenate([np.eye(size)[states[:, j]]
                        for j, size in enumerate((4, 4, 2, 6))], axis=1)
    y = np.where(states[:, 2] == 0, states[:, 0], states[:, 1])
    return states, x, y


def state_id(a, b, s, n):
    return ((a * 4 + b) * 2 + s) * 6 + n


def make_pairs(states):
    receiver, donor = np.indices((len(states), len(states)))
    r, d = receiver.ravel(), donor.ravel()
    different = states[r, 0] != states[d, 0]
    r, d = r[different], d[different]
    active = states[r, 2] == 0
    sm = states[r, 2] == states[d, 2]
    bm = states[r, 1] == states[d, 1]
    stratum = states[r, 2] * 4 + sm.astype(int) * 2 + bm.astype(int)
    test = np.zeros(len(r), dtype=bool)
    for group in range(8):
        ix = np.flatnonzero(stratum == group)
        assert len(ix) % 2 == 0
        ordered = sorted(ix, key=lambda i: hashlib.sha256(
            f"inactive-variable-audit-v1:{r[i]}:{d[i]}".encode()).digest())
        test[ordered[len(ordered)//2:]] = True
    cf = state_id(states[d, 0], states[r, 1], states[r, 2], states[r, 3])
    matched = sm & bm & (states[r, 3] == states[d, 3])
    return dict(r=r, d=d, active=active, stratum=stratum, test=test, cf=cf,
                matched=matched)


def make_same_pairs(states):
    r, d = np.indices((len(states), len(states)))
    r, d = r.ravel(), d.ravel()
    same = states[r, 0] == states[d, 0]
    return r[same], d[same]


def initialize(seed):
    rng = np.random.default_rng(seed)
    params = {}
    for i, (ni, no) in enumerate(((16, 12), (12, 24), (24, 4)), 1):
        params[f"W{i}"] = rng.normal(0, 1/np.sqrt(ni), (ni, no))
        params[f"b{i}"] = np.zeros(no)
    return params


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=1, keepdims=True)


def downstream(h1, p):
    h2 = np.tanh(h1 @ p["W2"] + p["b2"])
    return softmax(h2 @ p["W3"] + p["b3"])


def forward(x, p):
    h1 = np.tanh(x @ p["W1"] + p["b1"])
    h2 = np.tanh(h1 @ p["W2"] + p["b2"])
    logits = h2 @ p["W3"] + p["b3"]
    return softmax(logits), h1, h2, logits


def loss_grad(x, y, p):
    prob, h1, h2, logits = forward(x, p)
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    dz = prob.copy()
    dz[np.arange(len(y)), y] -= 1
    dz /= len(y)
    g = {"W3": h2.T @ dz, "b3": dz.sum(axis=0)}
    dh2 = (dz @ p["W3"].T) * (1-h2*h2)
    g.update(W2=h1.T @ dh2, b2=dh2.sum(axis=0))
    dh1 = (dh2 @ p["W2"].T) * (1-h1*h1)
    g.update(W1=x.T @ dh1, b1=dh1.sum(axis=0))
    return float(loss), g


def train(seed, x, y):
    p = initialize(seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(value) for k, value in p.items()}
    trace = []
    for step in range(1, STEPS+1):
        loss, grad = loss_grad(x, y, p)
        for name in p:
            m[name] = .9*m[name] + .1*grad[name]
            v[name] = .999*v[name] + .001*grad[name]*grad[name]
            p[name] -= LR * (m[name]/(1-.9**step)) / (
                np.sqrt(v[name]/(1-.999**step)) + 1e-8)
        if step == 1 or step % 100 == 0 or step == STEPS:
            after = forward(x, p)[0]
            trace.append(dict(seed=seed, step=step, loss_before_update=loss,
                              accuracy_after_update=float(np.mean(after.argmax(1)==y))))
    return p, trace


def patch(h1, p, r, d, mask):
    mixed = h1[r].copy()
    if mask:
        mixed[:, mask] = h1[d][:, mask]
    return downstream(mixed, p)


def tv(a, b):
    return .5*np.abs(a-b).sum(axis=1)


def execution_checks(x, states, pairs, p):
    prob, h1, _, _ = forward(x, p)
    r, d = pairs["r"], pairs["d"]
    # Full-pair tests check broadcasting and the last partial batch as well.
    errors = {
        "empty_patch_max_error": float(np.abs(patch(h1,p,r,d,())-prob[r]).max()),
        "identity_patch_max_error": float(np.abs(patch(h1,p,r,r,(0,2,7))-prob[r]).max()),
        "full_patch_max_error": float(np.abs(patch(h1,p,r,d,tuple(range(12)))-prob[d]).max()),
    }
    cf_states = states[r].copy()
    cf_states[:, 0] = states[d, 0]
    cf_x = np.concatenate([np.eye(size)[cf_states[:, j]]
                           for j,size in enumerate((4,4,2,6))], axis=1)
    errors["counterfactual_cache_max_error"] = float(np.abs(
        forward(cf_x,p)[0]-prob[pairs["cf"]]).max())
    assert max(errors.values()) < TOL, errors
    return errors


def correctness_checks():
    states, x, y = make_inputs()
    pairs = make_pairs(states)
    assert len(states) == 192 and x.shape == (192,16)
    assert len(pairs["r"]) == 27648
    assert pairs["test"].sum() == 13824
    assert pairs["matched"].sum() == 576
    for g in range(8):
        loc = pairs["stratum"] == g
        assert (loc & pairs["test"]).sum() == loc.sum()//2
    # Each ordered pair is unique, so discovery and test cannot overlap.
    ids = pairs["r"]*192 + pairs["d"]
    assert len(np.unique(ids)) == len(ids)
    same_r, same_d = make_same_pairs(states)
    assert len(same_r) == 9216
    assert np.all(states[same_r,0] == states[same_d,0])
    assert set(ids).isdisjoint(set(same_r*192+same_d))
    p = initialize(314)
    rng = np.random.default_rng(2718)
    batch = np.array([0,7,28,61,99,130,177,191])
    _, grad = loss_grad(x[batch],y[batch],p)
    locations = [(key,ix) for key,val in p.items() for ix in np.ndindex(val.shape)]
    chosen = rng.choice(len(locations),size=24,replace=False)
    max_grad_error = 0.0
    for loc in chosen:
        key, ix = locations[loc]
        original = p[key][ix]
        p[key][ix] = original+1e-5
        plus = loss_grad(x[batch],y[batch],p)[0]
        p[key][ix] = original-1e-5
        minus = loss_grad(x[batch],y[batch],p)[0]
        p[key][ix] = original
        estimate = (plus-minus)/2e-5
        error = abs(estimate-grad[key][ix])/max(1.,abs(estimate),abs(grad[key][ix]))
        max_grad_error = max(max_grad_error,float(error))
    assert max_grad_error < 1e-6, max_grad_error
    controls = execution_checks(x,states,pairs,p)
    # Exact factorized reference: only the a feature is replaced.
    r, d = pairs["r"],pairs["d"]
    reference_patch = np.where(states[r,2]==0,states[d,0],states[r,1])
    assert np.array_equal(reference_patch,y[pairs["cf"]])
    assert np.array_equal(reference_patch[~pairs["active"]],y[r][~pairs["active"]])
    for split in (False,True):
        for active in (False,True):
            assert np.any(pairs["matched"] & (pairs["test"]==split) &
                          (pairs["active"]==active))
    return dict(passed=True, gradient_coordinates=24,gradient_max_scaled_error=max_grad_error,
                pairs=27648, discovery_pairs=13824,test_pairs=13824,
                matched_pairs=576,same_a_pairs=9216,factorized_reference_passed=True,
                **controls)


def build_masks():
    return [(k, mask) for k in (1,2,3) for mask in itertools.combinations(range(12),k)]


def mean_where(values, keep):
    return float(values[keep].mean()) if np.any(keep) else None


def metrics(errors, correct, copies, pairs, keep, changes, intact_changes):
    active = pairs["active"]
    result = dict(n=int(keep.sum()), tv=mean_where(errors,keep),
                  active_tv=mean_where(errors,keep&active),
                  inactive_tv=mean_where(errors,keep&~active),
                  semantic_accuracy=mean_where(correct,keep),
                  donor_copy_rate=mean_where(copies,keep),
                  inactive_output_change_tv=mean_where(changes,keep&~active),
                  intact_inactive_change_tv=mean_where(intact_changes,keep&~active))
    result["worst_tv"] = max(result["active_tv"],result["inactive_tv"])
    for group in range(8):
        result[f"stratum_{group}_tv"] = mean_where(errors,keep&(pairs["stratum"]==group))
    return result


def choose(rows, key):
    best = min(row[key] for row in rows)
    # Input rows are already in lexicographic mask order.
    return next(row for row in rows if row[key] <= best+TOL)


def evaluate(seed, p, states, x, y, pairs, out):
    prob,h1,_,_ = forward(x,p)
    eligible = bool(np.mean(prob.argmax(1)==y) >= .99)
    r,d = pairs["r"],pairs["d"]
    target = prob[pairs["cf"]]
    intact_changes = tv(target,prob[r])
    semantic = y[pairs["cf"]]
    groups = {}
    for regime in ("unrestricted","matched"):
        valid = np.ones(len(r),dtype=bool) if regime=="unrestricted" else pairs["matched"]
        for split in ("discovery","test"):
            groups[(regime,split)] = valid & (pairs["test"] == (split=="test"))
    all_rows = []
    masks = build_masks()
    for mask_id,(k,mask) in enumerate(masks):
        prediction = patch(h1,p,r,d,mask)
        errors = tv(prediction,target)
        guesses = prediction.argmax(1)
        correct, copies = guesses==semantic, guesses==y[d]
        for (regime,split),keep in groups.items():
            all_rows.append(dict(seed=seed,eligible=eligible,mask_id=mask_id,k=k,
                                 mask=" ".join(map(str,mask)),regime=regime,split=split,
                                 **metrics(errors,correct,copies,pairs,keep,
                                           tv(prediction,prob[r]),intact_changes)))
    write_csv(out/f"candidate-scores-seed-{seed}.csv",all_rows)
    by_key = {(z["regime"],z["split"],z["mask_id"]):z for z in all_rows}
    selected = []
    for regime in ("unrestricted","matched"):
        for k in (1,2,3):
            discovery = [z for z in all_rows if z["regime"]==regime and
                         z["split"]=="discovery" and z["k"]==k]
            picks = [("repair_only", choose(discovery,"active_tv")),
                     ("balanced",choose(discovery,"worst_tv"))]
            random_rng = np.random.default_rng(9000+k)
            random_indices = sorted(random_rng.choice(len(discovery),
                                       size=min(20,len(discovery)),replace=False))
            picks += [(f"random_{j:02d}",discovery[ix]) for j,ix in enumerate(random_indices)]
            for method,disc in picks:
                test = by_key[(regime,"test",disc["mask_id"])]
                selected.append(dict(seed=seed,eligible=eligible,regime=regime,k=k,
                                     method=method,mask_id=disc["mask_id"],mask=disc["mask"],
                                     discovery_active_tv=disc["active_tv"],
                                     discovery_inactive_tv=disc["inactive_tv"],
                                     discovery_worst_tv=disc["worst_tv"],
                                     **{"test_"+key:value for key,value in test.items()
                                        if key in ("n","tv","active_tv","inactive_tv","worst_tv",
                                                   "semantic_accuracy","donor_copy_rate",
                                                   "inactive_output_change_tv","intact_inactive_change_tv")}))
    # The empty baseline is evaluated on exactly the same target and pair banks.
    base_errors = tv(prob[r],target)
    baselines = [dict(seed=seed,eligible=eligible,regime=regime,split=split,
                      **metrics(base_errors,prob[r].argmax(1)==semantic,
                                prob[r].argmax(1)==y[d],pairs,keep,
                                np.zeros(len(r)),intact_changes))
                 for (regime,split),keep in groups.items()]
    same_r,same_d = make_same_pairs(states)
    same_controls = []
    unique_mask_ids = sorted({z["mask_id"] for z in selected if not z["method"].startswith("random_")})
    for mask_id in unique_mask_ids:
        k,mask = masks[mask_id]
        prediction = patch(h1,p,same_r,same_d,mask)
        error = tv(prediction,prob[same_r])
        active = states[same_r,2]==0
        same_controls.append(dict(seed=seed,eligible=eligible,mask_id=mask_id,k=k,
                                  mask=" ".join(map(str,mask)),n=len(same_r),
                                  tv=float(error.mean()),active_tv=float(error[active].mean()),
                                  inactive_tv=float(error[~active].mean()),
                                  changed_answer_rate=float(np.mean(prediction.argmax(1)!=prob[same_r].argmax(1)))))
    return all_rows,selected,baselines,same_controls


def bootstrap_summary(values):
    values = np.array(values,dtype=float)
    if len(values)==0:
        return dict(n=0,mean=None,median=None,minimum=None,maximum=None,ci95=None)
    result = dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                  minimum=float(values.min()),maximum=float(values.max()),ci95=None)
    if len(values)>=6:
        rng = np.random.default_rng(20261013)
        samples = rng.choice(values,size=(10000,len(values)),replace=True).mean(axis=1)
        result["ci95"] = list(map(float,np.quantile(samples,[.025,.975])))
    return result


def verify_lock(commit):
    if not commit:
        raise ValueError("--run requires --lock-commit. Commit the code and plan first.")
    resolved = subprocess.check_output(["git","rev-parse",commit+"^{commit}"],cwd=REPO,text=True).strip()
    records = {}
    for file in (STUDY/"analysis-plan.md",Path(__file__).resolve()):
        relative = file.relative_to(REPO).as_posix()
        locked = subprocess.check_output(["git","show",resolved+":"+relative],cwd=REPO)
        current = file.read_bytes()
        if locked != current:
            raise ValueError("File differs from lock commit: "+relative)
        records[relative] = digest_bytes(current)
    return dict(commit=resolved,sha256=records)


def run(lock_commit):
    lock = verify_lock(lock_commit)
    started = time.monotonic()
    checks = correctness_checks()
    states,x,y = make_inputs()
    pairs = make_pairs(states)
    results = STUDY/"results"
    models = STUDY/"models"
    if results.exists() and any(results.iterdir()):
        raise ValueError("Results already exist. Move them explicitly before a fresh run.")
    results.mkdir(parents=True,exist_ok=True)
    models.mkdir(parents=True,exist_ok=True)
    write_json(results/"lock.json",lock)
    write_json(results/"preflight-controls.json",checks)
    write_csv(STUDY/"data"/"inputs.csv",[
        dict(id=i,a=int(z[0]),b=int(z[1]),s=int(z[2]),n=int(z[3]),label=int(y[i]))
        for i,z in enumerate(states)])
    write_csv(STUDY/"data"/"pairs.csv",[
        dict(pair_id=int(r*192+d),receiver=int(r),donor=int(d),
             counterfactual=int(pairs["cf"][i]),stratum=int(pairs["stratum"][i]),
             split="test" if pairs["test"][i] else "discovery",
             matched=bool(pairs["matched"][i]))
        for i,(r,d) in enumerate(zip(pairs["r"],pairs["d"]))])
    seed_rows,selected_rows,baseline_rows,same_rows,loss_rows,prediction_rows = [],[],[],[],[],[]
    for seed in SEEDS:
        seed_started = time.monotonic()
        p,trace = train(seed,x,y)
        np.savez(models/f"seed-{seed}.npz",**p)
        loss_rows.extend(trace)
        controls = execution_checks(x,states,pairs,p)
        write_json(results/f"controls-seed-{seed}.json",controls)
        prob = forward(x,p)[0]
        accuracy = float(np.mean(prob.argmax(1)==y))
        _,selected,baselines,same = evaluate(seed,p,states,x,y,pairs,results)
        selected_rows.extend(selected)
        baseline_rows.extend(baselines)
        same_rows.extend(same)
        for i,row in enumerate(prob):
            prediction_rows.append(dict(seed=seed,input_id=i,label=int(y[i]),
                                        prediction=int(row.argmax()),
                                        **{f"p{j}":float(row[j]) for j in range(4)}))
        repair = next(z for z in selected if z["regime"]=="unrestricted" and z["k"]==3 and z["method"]=="repair_only")
        balance = next(z for z in selected if z["regime"]=="unrestricted" and z["k"]==3 and z["method"]=="balanced")
        seed_rows.append(dict(seed=seed,eligible=accuracy>=.99,accuracy=accuracy,
                              final_loss=loss_grad(x,y,p)[0],
                              repair_mask=repair["mask"],balanced_mask=balance["mask"],
                              repair_active_tv=repair["test_active_tv"],
                              repair_inactive_tv=repair["test_inactive_tv"],
                              repair_worst_tv=repair["test_worst_tv"],
                              balanced_active_tv=balance["test_active_tv"],
                              balanced_inactive_tv=balance["test_inactive_tv"],
                              balanced_worst_tv=balance["test_worst_tv"],
                              improvement=repair["test_worst_tv"]-balance["test_worst_tv"],
                              seconds=time.monotonic()-seed_started))
        write_csv(results/"seed-outcomes.csv",seed_rows)
        print(json.dumps(seed_rows[-1]),flush=True)
    write_csv(results/"selected-masks.csv",selected_rows)
    write_csv(results/"empty-baselines.csv",baseline_rows)
    write_csv(results/"same-a-controls.csv",same_rows)
    write_csv(results/"training-trace.csv",loss_rows)
    write_csv(results/"intact-predictions.csv",prediction_rows)
    main = bootstrap_summary([z["improvement"] for z in seed_rows if z["eligible"]])
    summary = dict(question="Does balanced selection reduce held-out worst-group error?",
                   primary_size=3,primary_regime="unrestricted",positive_difference="favours balanced",
                   eligible_seed_count=main["n"],gate_passed=main["n"]>=6,
                   primary=main,all_seeds=seed_rows,
                   elapsed_seconds=time.monotonic()-started,
                   environment=dict(python=platform.python_version(),numpy=np.__version__,
                                    platform=platform.platform(),blas_threads=1),
                   lock=lock)
    write_json(results/"summary.json",summary)
    # Recheck code and plan at completion, then hash every scientific output.
    assert verify_lock(lock_commit)==lock
    files = sorted(list((STUDY/"data").glob("*"))+list(models.glob("*"))+list(results.glob("*")))
    write_json(results/"manifest.json",{f.relative_to(STUDY).as_posix():digest_bytes(f.read_bytes())
                                        for f in files if f.name!="manifest.json"})
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check",action="store_true")
    modes.add_argument("--run",action="store_true")
    parser.add_argument("--lock-commit")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(correctness_checks(),indent=2))
    else:
        run(args.lock_commit)


if __name__=="__main__":
    main()

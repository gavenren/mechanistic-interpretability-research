"""Three-direction edit comparison. Importing this file runs no model."""
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

SEEDS = tuple(range(91001, 91009))
MASKS = tuple(itertools.combinations(range(12), 3))
RANDOM_IDS = np.random.Generator(np.random.PCG64(94001)).permutation(220)[:20]
STEPS = 4000
CHECKPOINTS = (1000, 2000, 3000, 4000)
TOL = 1e-10
BATCH = 768


def _models_module():
    spec = importlib.util.spec_from_file_location("specificity_base_models", Path(__file__).with_name("models.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _empty(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Edit output folder is not empty; preserve it and stop.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _close(a, b, name, tol=TOL):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Edit shape or finite-value check failed: " + name)
    error = float(np.max(np.abs(a - b)))
    if error > tol:
        raise RuntimeError(f"Edit replay failed: {name}: {error}")
    return error


def _save(path, **arrays):
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError("Refuse a nonfinite saved edit array: " + key)
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise RuntimeError("Saved edit array names changed; stop.")
        saved = {key: archive[key].copy() for key in archive.files}
    for key, value in arrays.items():
        if saved[key].dtype != value.dtype or not np.array_equal(saved[key], value):
            raise RuntimeError("Saved edit array changed: " + key)
    return saved


def _affine(x, w, b, reference=False):
    if not reference:
        return x @ w + b
    result = np.broadcast_to(b, (len(x), len(b))).copy()
    for coordinate in range(x.shape[1]):
        result += x[:, coordinate, None] * w[coordinate, None, :]
    return result


def _softmax(logits, reference=False):
    ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    if not reference:
        return ex / ex.sum(axis=1, keepdims=True)
    total = np.zeros(len(ex))
    for coordinate in range(ex.shape[1]):
        total += ex[:, coordinate]
    return ex / total[:, None]


def _downstream(h, p, reference=False):
    hidden = np.tanh(_affine(h, p["W2"], p["b2"], reference))
    logits = _affine(hidden, p["W3"], p["b3"], reference)
    return logits, _softmax(logits, reference)


def _context(x, p):
    h = np.tanh(_affine(x, p["W1"], p["b1"]))
    hr = np.tanh(_affine(x, p["W1"], p["b1"], True))
    logits, probabilities = _downstream(h, p)
    lr, pr = _downstream(hr, p, True)
    errors = [_close(h, hr, "intact hidden"), _close(logits, lr, "intact logits"),
              _close(probabilities, pr, "intact probabilities")]
    pt = {key: torch.from_numpy(value.copy()) for key, value in p.items()}
    with torch.no_grad():
        ht = torch.tanh(torch.from_numpy(x) @ pt["W1"] + pt["b1"])
        lt = torch.tanh(ht @ pt["W2"] + pt["b2"]) @ pt["W3"] + pt["b3"]
        qt = torch.softmax(lt, dim=1)
    errors += [_close(ht.numpy(), h, "Torch hidden"), _close(lt.numpy(), logits, "Torch logits"),
               _close(qt.numpy(), probabilities, "Torch probabilities")]
    return dict(h=h, hr=hr, logits=logits, probabilities=probabilities, maximum_intact_error=max(errors))


def _load_data(models_path):
    models_path = Path(models_path)
    with np.load(models_path / "data.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    module = _models_module()
    states, x, labels = module.inputs()
    expected = dict(states=states, x=x, labels=labels, **module.pairs())
    if set(data) != set(expected):
        raise RuntimeError("Fixed edit data keys changed; stop.")
    for key in expected:
        if not np.array_equal(data[key], expected[key]):
            raise RuntimeError("Fixed edit input ledger changed: " + key)
    summary = json.loads((models_path / "summary.json").read_text())
    if summary.get("ability_gate_pass") is not True:
        raise RuntimeError("All eight models must pass the ability gate before edit work.")
    return data


def _load_model(models_path, seed, data):
    path = Path(models_path) / f"seed-{seed}/final.npz"
    with np.load(path, allow_pickle=False) as archive:
        p = {key: archive[key].copy() for key in archive.files}
    shapes = {"W1": (16, 12), "b1": (12,), "W2": (12, 24), "b2": (24,), "W3": (24, 4), "b3": (4,)}
    if set(p) != set(shapes) or any(p[k].shape != shapes[k] or p[k].dtype != np.float64 or not np.isfinite(p[k]).all() for k in shapes):
        raise RuntimeError("Base-model weights or shapes are invalid; stop.")
    context = _context(data["x"], p)
    accuracy = float(np.mean(context["probabilities"].argmax(1) == data["labels"]))
    if accuracy < .99:
        raise RuntimeError("A base model fails the fixed 99 percent gate; stop.")
    return p, context, dict(seed=seed, weight_sha256=_hash(path), intact_accuracy=accuracy,
                            maximum_intact_error=context["maximum_intact_error"])


def _expm_reference(skew):
    norm = float(np.max(np.sum(np.abs(skew), axis=0)))
    if not np.isfinite(norm):
        raise RuntimeError("Nonfinite rotation parameter; stop.")
    scale = max(0, int(np.ceil(np.log2(norm / .5)))) if norm > .5 else 0
    if scale > 16:
        raise RuntimeError("Rotation exceeds fixed reference range; stop.")
    small = skew / (2. ** scale)
    result = np.eye(12)
    term = np.eye(12)
    for degree in range(1, 41):
        term = term @ small / degree
        result = result + term
    for _ in range(scale):
        result = result @ result
    return result


def _check_rotation(a, q):
    skew = a - a.T
    reference = _expm_reference(skew)
    return max(_close(q, reference, "matrix exponential reference"),
               _close(q.T @ q, np.eye(12), "rotation orthogonality"),
               _close(q @ q.T, np.eye(12), "reverse rotation orthogonality"))


def _mixed(kind, ids, data, hidden, mask=None, rotation=None, reference=False):
    r, d = data["r"][ids], data["d"][ids]
    if kind == "coherent":
        return hidden[data["cf"][ids]].copy()
    if kind == "full":
        return hidden[d].copy()
    result = hidden[r].copy()
    if kind == "none":
        return result
    if kind == "mask":
        if reference:
            for coordinate in mask:
                result[:, coordinate] = hidden[d, coordinate]
        else:
            result[:, mask] = hidden[d][:, mask]
        return result
    if kind != "rotation":
        raise RuntimeError("Unknown edit kind; stop.")
    if not reference:
        basis = rotation[:, :3]
        return result + ((hidden[d] - result) @ basis) @ basis.T
    # Separate full rotate, replace, and undo implementation.
    rotated = _affine(result, rotation, np.zeros(12), True)
    donor_rotated = _affine(hidden[d], rotation, np.zeros(12), True)
    rotated[:, :3] = donor_rotated[:, :3]
    return _affine(rotated, rotation.T, np.zeros(12), True)


def _metrics(probabilities, ids, data, context):
    active = data["active"][ids].astype(bool)
    same = data["same_value"][ids].astype(bool)
    r, cf = data["r"][ids], data["cf"][ids]
    target = context["probabilities"][cf]
    errors = .5 * np.sum(np.abs(probabilities - target), axis=1)
    receiver_errors = .5 * np.sum(np.abs(probabilities - context["probabilities"][r]), axis=1)
    answer = probabilities.argmax(1)
    correct = answer == data["labels"][cf]
    changes = answer != context["probabilities"][r].argmax(1)
    groups = {"active_changed": active & ~same, "inactive_changed": ~active & ~same,
              "all_changed": ~same, "same_value": same}
    result = {}
    for name, group in groups.items():
        if not np.any(group):
            raise RuntimeError("A required edit group is empty; stop.")
        result[name + "_n"] = int(group.sum())
        result[name + "_accuracy"] = float(np.mean(correct[group]))
        result[name + "_tv"] = float(np.mean(errors[group]))
        result[name + "_receiver_tv"] = float(np.mean(receiver_errors[group]))
        result[name + "_answer_change"] = float(np.mean(changes[group]))
    result["eligible"] = result["inactive_changed_tv"] <= .05 and result["same_value_answer_change"] <= .01
    result["absolute_gate_pass"] = bool(result["active_changed_accuracy"] >= .95 and result["all_changed_accuracy"] >= .95
                                          and result["active_changed_tv"] <= .05 and result["inactive_changed_tv"] <= .05
                                          and result["same_value_answer_change"] <= .01)
    return result


def _evaluate(kind, ids, data, p, context, guard, mask=None, rotation=None, save_path=None):
    logits = np.empty((len(ids), 4), np.float64)
    probabilities = np.empty_like(logits)
    reference_probabilities = np.empty_like(logits)
    replay_error = 0.
    for start in range(0, len(ids), BATCH):
        guard()
        subset = ids[start:start + BATCH]
        state = _mixed(kind, subset, data, context["h"], mask, rotation)
        reference_state = _mixed(kind, subset, data, context["hr"], mask, rotation, True)
        values, probs = _downstream(state, p)
        lr, pr = _downstream(reference_state, p, True)
        replay_error = max(replay_error, _close(state, reference_state, "edited state"),
                           _close(values, lr, "edited logits"), _close(probs, pr, "edited probabilities"))
        if kind in ("none", "full", "coherent"):
            target_id = data[{"none": "r", "full": "d", "coherent": "cf"}[kind]][subset]
            replay_error = max(replay_error, _close(values, context["logits"][target_id], "control logits"),
                               _close(probs, context["probabilities"][target_id], "control probabilities"))
        logits[start:start + len(subset)], probabilities[start:start + len(subset)] = values, probs
        reference_probabilities[start:start + len(subset)] = pr
    metrics = _metrics(probabilities, ids, data, context)
    reference_metrics = _metrics(reference_probabilities, ids, data, context)
    if not np.array_equal(probabilities.argmax(1), reference_probabilities.argmax(1)):
        raise RuntimeError("Independent edited answer differs; stop.")
    metric_error = 0.
    for key in metrics:
        if isinstance(metrics[key], (bool, int)):
            if metrics[key] != reference_metrics[key]:
                raise RuntimeError("Independent metric gate differs: " + key)
        else:
            metric_error = max(metric_error, _close(metrics[key], reference_metrics[key], "metric " + key))
    if save_path is not None:
        saved = _save(save_path, pair_ids=ids.astype(np.int64), logits=logits, probabilities=probabilities)
        for start in range(0, len(ids), BATCH):
            guard()
            subset = saved["pair_ids"][start:start + BATCH]
            reference_state = _mixed(kind, subset, data, context["hr"], mask, rotation, True)
            lr, pr = _downstream(reference_state, p, True)
            replay_error = max(replay_error, _close(saved["logits"][start:start + len(subset)], lr, "saved logits"),
                               _close(saved["probabilities"][start:start + len(subset)], pr, "saved probabilities"))
        saved_metrics = _metrics(saved["probabilities"], saved["pair_ids"], data, context)
        if saved_metrics != metrics:
            raise RuntimeError("Saved edit metrics differ; stop.")
    return dict(**metrics, maximum_replay_error=replay_error, maximum_metric_replay_error=metric_error)


def _learning_setup(data, context, p, ids):
    active = data["active"][ids].astype(bool)
    same = data["same_value"][ids].astype(bool)
    groups = (ids[active & ~same], ids[~active & ~same], ids[same])
    if any(len(group) == 0 for group in groups):
        raise RuntimeError("A learning group is empty; stop.")
    return groups, torch.from_numpy(context["h"].copy()), torch.from_numpy(context["probabilities"].copy()), {
        key: torch.from_numpy(p[key].copy()) for key in ("W2", "b2", "W3", "b3")}


def _loss(a, pair_ids, data, hidden, target, p):
    q = torch.matrix_exp(a - a.T)
    basis = q[:, :3]
    r = torch.from_numpy(data["r"][pair_ids].astype(np.int64))
    d = torch.from_numpy(data["d"][pair_ids].astype(np.int64))
    cf = torch.from_numpy(data["cf"][pair_ids].astype(np.int64))
    edited = hidden[r] + ((hidden[d] - hidden[r]) @ basis) @ basis.T
    logits = torch.tanh(edited @ p["W2"] + p["b2"]) @ p["W3"] + p["b3"]
    tv = .5 * torch.abs(torch.softmax(logits, dim=1) - target[cf]).sum(dim=1)
    group_loss = tv.reshape(3, 64).mean(dim=1)
    return group_loss.sum(), group_loss


def _train_rotation(seed, out, data, p, context, guard, steps=STEPS):
    ids = np.flatnonzero(data["split"] == 0)
    groups, hidden, target, weights = _learning_setup(data, context, p, ids)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    a = torch.nn.Parameter(.01 * torch.randn((12, 12), generator=generator, dtype=torch.float64))
    initial = a.detach().numpy().copy()
    optimizer = torch.optim.Adam([a], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                                 amsgrad=False, foreach=False, fused=False, maximize=False,
                                 capturable=False, differentiable=False)
    rng = np.random.Generator(np.random.PCG64(seed + 2000))
    draws = np.empty((steps, 192), dtype=np.int64)
    for step in range(steps):
        for group in range(3):
            draws[step, 64 * group:64 * (group + 1)] = groups[group][rng.integers(0, len(groups[group]), size=64)]
    if out is not None:
        _save(out / "learning-draws.npz", pair_ids=draws, initial_a=initial)
    losses = np.empty((steps, 4), dtype=np.float64)
    checkpoints = {}
    # Replace this tuple only after all parts of an update pass their checks.
    # A failure inside optimizer.step must not relabel a partial update.
    completed_state = (0, initial.copy())
    started = time.monotonic()
    try:
        for step in range(1, steps + 1):
            guard()
            optimizer.zero_grad(set_to_none=True)
            loss, group_loss = _loss(a, draws[step - 1], data, hidden, target, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite rotation loss; stop.")
            loss.backward()
            if a.grad is None or not torch.isfinite(a.grad).all():
                raise RuntimeError("Nonfinite rotation gradient; stop.")
            optimizer.step()
            if not torch.isfinite(a).all():
                raise RuntimeError("Nonfinite rotation parameter; stop.")
            adam_state = optimizer.state[a]
            if not torch.isfinite(adam_state["exp_avg"]).all() or not torch.isfinite(adam_state["exp_avg_sq"]).all():
                raise RuntimeError("Nonfinite Adam moment; stop.")
            if not torch.isfinite(adam_state["step"]).all():
                raise RuntimeError("Nonfinite Adam step counter; stop.")
            losses[step - 1] = np.r_[float(loss.detach()), group_loss.detach().numpy()]
            completed_state = (step, a.detach().numpy().copy())
            if step in CHECKPOINTS or step == steps:
                with torch.no_grad():
                    q = torch.matrix_exp(a - a.T).numpy()
                parameter = a.detach().numpy().copy()
                error = _check_rotation(parameter, q)
                checkpoints[step] = dict(a=parameter, q=q.copy(), reference_error=error)
                if out is not None:
                    state = optimizer.state[a]
                    saved = _save(out / f"rotation-step-{step}.npz", a=parameter, q=q,
                                  adam_mean=state["exp_avg"].numpy(), adam_variance=state["exp_avg_sq"].numpy(),
                                  adam_step=np.array(step, dtype=np.int64), losses=losses[:step].copy())
                    _check_rotation(saved["a"], saved["q"])
        return checkpoints, dict(updates=completed_state[0], elapsed_seconds=time.monotonic() - started,
                                 initialization_seed=seed + 1000, minibatch_seed=seed + 2000)
    except BaseException:
        if out is not None:
            completed_step, completed_a = completed_state
            _save(out / "rotation-partial.npz", a=completed_a, losses=losses[:completed_step].copy(),
                  completed_updates=np.array(completed_step, dtype=np.int64),
                  known_complete_update_boundary=np.array(True, dtype=np.bool_))
        raise


def _source_hashes(models_path):
    root = Path(models_path)
    paths = ["data.npz", "summary.json"] + [f"seed-{seed}/final.npz" for seed in SEEDS]
    return {path: _hash(root / path) for path in paths}


def _verify_sources(models_path, expected):
    if _source_hashes(models_path) != expected:
        raise RuntimeError("Published base-model files changed; stop.")


def _all_models(models_path, data, guard):
    models = {}
    for seed in SEEDS:
        guard()
        models[seed] = _load_model(models_path, seed, data)
    return models


def _manifest(out):
    files = {str(path.relative_to(out)): _hash(path) for path in sorted(out.rglob("*"))
             if path.is_file() and path.name != "calibration-manifest.json"}
    _json(out / "calibration-manifest.json", dict(files=files))


def _verify_calibration(path, guard):
    path = Path(path)
    manifest = json.loads((path / "calibration-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        guard()
        if _hash(path / name) != expected:
            raise RuntimeError("Published edit calibration changed: " + name)


def _checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        a, q = archive["a"].copy(), archive["q"].copy()
    _check_rotation(a, q)
    return q


def prepare(out, guard):
    """Dummy arithmetic, gradient controls, and bounded timing only."""
    started = time.monotonic()
    out = _empty(out)
    guard()
    module = _models_module()
    states, x, labels = module.inputs()
    data = dict(states=states, x=x, labels=labels, **module.pairs())
    rng = np.random.Generator(np.random.PCG64(95001))
    p = {}
    for layer, ni, no in ((1, 16, 12), (2, 12, 24), (3, 24, 4)):
        p[f"W{layer}"] = rng.normal(0., 1. / np.sqrt(ni), (ni, no))
        p[f"b{layer}"] = rng.normal(0., .01, no)
    context = _context(x, p)
    selection_ids = np.flatnonzero(data["split"] == 1)
    test_ids = np.flatnonzero(data["split"] == 2)
    first = selection_ids[:BATCH]
    identity = np.eye(12)
    identity_error = _close(_mixed("rotation", first, data, context["h"], rotation=identity),
                            _mixed("mask", first, data, context["h"], mask=(0, 1, 2)), "identity rotation")
    quarter = np.eye(12)
    quarter[0, 0] = quarter[3, 3] = 0.
    quarter[0, 3], quarter[3, 0] = -1., 1.
    quarter_error = _close(_mixed("rotation", first, data, context["h"], rotation=quarter),
                           _mixed("mask", first, data, context["h"], mask=(1, 2, 3)), "known coordinate rotation")
    groups, hidden, target, weights = _learning_setup(data, context, p, np.flatnonzero(data["split"] == 0))
    sample = np.concatenate([group[:64] for group in groups])
    generator = torch.Generator(device="cpu").manual_seed(95002)
    a = torch.nn.Parameter(.01 * torch.randn((12, 12), generator=generator, dtype=torch.float64))
    loss, _ = _loss(a, sample, data, hidden, target, weights)
    loss.backward()
    gradient = a.grad.detach().numpy().copy()
    errors = []
    for coordinate in ((0, 1), (1, 2), (3, 4), (10, 11)):
        original = float(a.detach()[coordinate])
        with torch.no_grad():
            a[coordinate] = original + 1e-6
        plus = float(_loss(a, sample, data, hidden, target, weights)[0].detach())
        with torch.no_grad():
            a[coordinate] = original - 1e-6
        minus = float(_loss(a, sample, data, hidden, target, weights)[0].detach())
        with torch.no_grad():
            a[coordinate] = original
        errors.append(abs((plus - minus) / 2e-6 - gradient[coordinate]))
    if max(errors) > 1e-5:
        raise RuntimeError("Rotation gradient control failed; stop.")
    for kind in ("none", "full", "coherent"):
        _evaluate(kind, selection_ids, data, p, context, guard)
    mask_started = time.monotonic()
    for mask in MASKS[:2]:
        _evaluate("mask", selection_ids, data, p, context, guard, mask=mask)
    mask_seconds = (time.monotonic() - mask_started) / 2.
    checkpoints, train = _train_rotation(91001, None, data, p, context, guard, steps=40)
    rotation = checkpoints[40]["q"]
    rotation_started = time.monotonic()
    _evaluate("rotation", selection_ids, data, p, context, guard, rotation=rotation)
    rotation_seconds = time.monotonic() - rotation_started
    saving_started = time.monotonic()
    _evaluate("rotation", test_ids, data, p, context, guard, rotation=rotation, save_path=out / "dummy-replay.npz")
    saving_seconds = time.monotonic() - saving_started
    estimate = 4. * (8. * (220. * mask_seconds + 4000. / 40. * train["elapsed_seconds"]
                           + 4. * rotation_seconds + 29. * saving_seconds) + 60.)
    report = dict(passed=True, dummy_only=True, identity_error=identity_error, quarter_turn_error=quarter_error,
                  gradient_maximum_error=max(errors), mask_seconds=mask_seconds,
                  dummy_40_update_seconds=train["elapsed_seconds"], rotation_selection_seconds=rotation_seconds,
                  final_saved_replay_seconds=saving_seconds, projected_seconds=estimate,
                  projection_rule="4 times eight models times (220 mask tests + 4000 rotation updates + four checkpoint tests + 29 saved tests), plus 4 times 60 seconds",
                  elapsed_seconds=time.monotonic() - started)
    _json(out / "edits-preparation.json", report)
    return report


def calibrate(out, guard, models_path):
    """Use learning and selection pairs only; parent must publish choices next."""
    started = time.monotonic()
    out = _empty(out)
    guard()
    hashes = _source_hashes(models_path)
    data = _load_data(models_path)
    models = _all_models(models_path, data, guard)
    selection_ids = np.flatnonzero(data["split"] == 1)
    choices, base_checks, training = {}, [], []
    for seed in SEEDS:
        guard()
        destination = out / f"seed-{seed}"
        destination.mkdir()
        p, context, base_check = models[seed]
        base_checks.append(base_check)
        scores = []
        with (destination / "mask-scores.csv").open("w", newline="") as handle:
            writer = None
            for index, mask in enumerate(MASKS):
                metric = _evaluate("mask", selection_ids, data, p, context, guard, mask=mask)
                row = dict(candidate=index, mask=" ".join(map(str, mask)), **metric)
                scores.append(row)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
        method1 = min(scores, key=lambda row: (row["active_changed_tv"], row["candidate"]))
        eligible = [row for row in scores if row["eligible"]]
        method2 = min(eligible, key=lambda row: (row["active_changed_tv"], row["candidate"])) if eligible else None
        checkpoints, training_record = _train_rotation(seed, destination, data, p, context, guard)
        training.append(dict(seed=seed, **training_record))
        rotation_scores = []
        for step in CHECKPOINTS:
            metric = _evaluate("rotation", selection_ids, data, p, context, guard, rotation=checkpoints[step]["q"])
            rotation_scores.append(dict(candidate=step, **metric))
        _csv(destination / "rotation-scores.csv", rotation_scores)
        eligible_rotation = [row for row in rotation_scores if row["eligible"]]
        method3 = min(eligible_rotation, key=lambda row: (row["active_changed_tv"], row["candidate"])) if eligible_rotation else None
        record = dict(seed=seed, weight_sha256=base_check["weight_sha256"], methods={})
        for name, selected in (("method1", method1), ("method2", method2), ("method3", method3)):
            if selected is None:
                record["methods"][name] = dict(available=False, reason="no candidate meets fixed selection limits")
                continue
            choice = dict(available=True, selection_metrics=selected)
            if name == "method3":
                step = selected["candidate"]
                checkpoint_path = f"seed-{seed}/rotation-step-{step}.npz"
                choice.update(kind="rotation", checkpoint=checkpoint_path, checkpoint_sha256=_hash(out / checkpoint_path), step=step)
                q = _checkpoint(out / checkpoint_path)
                raw = _evaluate("rotation", selection_ids, data, p, context, guard, rotation=q,
                                save_path=destination / f"{name}-selection.npz")
            else:
                mask = MASKS[selected["candidate"]]
                choice.update(kind="mask", mask=list(mask), candidate=selected["candidate"])
                raw = _evaluate("mask", selection_ids, data, p, context, guard, mask=mask,
                                save_path=destination / f"{name}-selection.npz")
            for key, value in raw.items():
                if key not in ("maximum_replay_error", "maximum_metric_replay_error") and value != selected[key]:
                    raise RuntimeError("Selected raw result differs from candidate score; stop.")
            choice["saved_selection_replay_error"] = raw["maximum_replay_error"]
            record["methods"][name] = choice
        choices[str(seed)] = record
        _json(destination / "choices.json", record)
        _verify_sources(models_path, hashes)
    _json(out / "choices.json", dict(source_hashes=hashes, models=choices, final_test_effects_collected=False,
                                     normal_significance_test_used=False, candidate_masks=[list(mask) for mask in MASKS],
                                     random_control_mask_ids=RANDOM_IDS.tolist()))
    _json(out / "base-checks.json", base_checks)
    _json(out / "training.json", training)
    summary = dict(passed=True, status="complete", selection_complete=True, final_test_effects_collected=False,
                   models=8, total_rotation_updates=8 * STEPS,
                   eligible_method2_models=sum(record["methods"]["method2"]["available"] for record in choices.values()),
                   eligible_method3_models=sum(record["methods"]["method3"]["available"] for record in choices.values()),
                   elapsed_seconds=time.monotonic() - started)
    _json(out / "summary.json", summary)
    _manifest(out)
    guard()
    return summary


def run(out, guard, models_path, calibration_path):
    """Run only after the parent verifies the public choice and code locks."""
    started = time.monotonic()
    out = _empty(out)
    calibration_path = Path(calibration_path)
    guard()
    _verify_calibration(calibration_path, guard)
    choices = json.loads((calibration_path / "choices.json").read_text())
    _verify_sources(models_path, choices["source_hashes"])
    data = _load_data(models_path)
    models = _all_models(models_path, data, guard)
    test_ids = np.flatnonzero(data["split"] == 2)
    _save(out / "test-ledger.npz", pair_ids=test_ids, r=data["r"][test_ids], d=data["d"][test_ids],
          cf=data["cf"][test_ids], active=data["active"][test_ids], same_value=data["same_value"][test_ids],
          required_label=data["labels"][data["cf"][test_ids]])
    rows, per_model, base_checks = [], [], []
    for seed in SEEDS:
        guard()
        p, context, base_check = models[seed]
        base_checks.append(base_check)
        destination = out / f"seed-{seed}"
        destination.mkdir()
        selected = choices["models"][str(seed)]["methods"]
        outcomes = {}
        for name in ("method1", "method2", "method3"):
            choice = selected[name]
            if not choice["available"]:
                outcomes[name] = dict(available=False, absolute_gate_pass=False, reason=choice["reason"])
                continue
            q, mask = None, None
            if choice["kind"] == "rotation":
                checkpoint = calibration_path / choice["checkpoint"]
                if _hash(checkpoint) != choice["checkpoint_sha256"]:
                    raise RuntimeError("Selected rotation file changed; stop.")
                q = _checkpoint(checkpoint)
            else:
                mask = tuple(choice["mask"])
                if len(mask) != 3 or mask != MASKS[choice["candidate"]]:
                    raise RuntimeError("Selected neuron mask changed; stop.")
            metric = _evaluate(choice["kind"], test_ids, data, p, context, guard, mask=mask, rotation=q,
                               save_path=destination / f"{name}-test.npz")
            outcomes[name] = dict(available=True, **metric)
            rows.append(dict(seed=seed, method=name, **metric))
        for kind in ("none", "full", "coherent"):
            metric = _evaluate(kind, test_ids, data, p, context, guard, save_path=destination / f"control-{kind}.npz")
            rows.append(dict(seed=seed, method="control_" + kind, **metric))
        for index, candidate in enumerate(RANDOM_IDS):
            metric = _evaluate("mask", test_ids, data, p, context, guard, mask=MASKS[candidate],
                               save_path=destination / f"control-random-{index:02d}.npz")
            rows.append(dict(seed=seed, method=f"random_{index:02d}_mask_{candidate}", **metric))
        model_record = dict(seed=seed, outcomes=outcomes)
        if outcomes["method3"]["available"]:
            model_record["active_accuracy_gain"] = outcomes["method3"]["active_changed_accuracy"] - outcomes["method1"]["active_changed_accuracy"]
            model_record["all_changed_accuracy_gain"] = outcomes["method3"]["all_changed_accuracy"] - outcomes["method1"]["all_changed_accuracy"]
        else:
            model_record["active_accuracy_gain"] = None
            model_record["all_changed_accuracy_gain"] = None
        per_model.append(model_record)
        _json(destination / "summary.json", model_record)
    _csv(out / "test-group-metrics.csv", rows)
    _json(out / "base-checks.json", base_checks)
    gates = {method: all(record["outcomes"][method]["available"] and record["outcomes"][method]["absolute_gate_pass"]
                         for record in per_model) for method in ("method1", "method2", "method3")}
    gains = [record["active_accuracy_gain"] for record in per_model]
    complete = all(value is not None for value in gains)
    mean_gain = float(np.mean(gains)) if complete else None
    positive = sum(value > 0. for value in gains if value is not None)
    improvement = bool(complete and gates["method3"] and mean_gain >= .05 and positive >= 6)
    _verify_sources(models_path, choices["source_hashes"])
    _verify_calibration(calibration_path, guard)
    summary = dict(passed=True, status="complete", numerical_controls_pass=True, models=8,
                   variable_explanation_gate_by_method=gates, learned_method_improvement_gate_pass=improvement,
                   primary_active_accuracy_mean_gain=mean_gain, positive_model_differences=positive,
                   all_changed_secondary_mean_gain=float(np.mean([record["all_changed_accuracy_gain"] for record in per_model])) if complete else None,
                   final_test_pairs_per_model=len(test_ids), normal_significance_test_used=False,
                   missing_eligible_candidates_are_failures_of_method_claim=True,
                   elapsed_seconds=time.monotonic() - started)
    _json(out / "summary.json", summary)
    _json(out / "per-model-results.json", per_model)
    guard()
    return summary

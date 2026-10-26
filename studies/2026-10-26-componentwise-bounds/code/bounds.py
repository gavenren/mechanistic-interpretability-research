"""Eight-term residual-bound comparison. Importing this module runs no model."""
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
OLD = STUDY.parent / "2026-10-13-equilibrium-sign-audit"
SEEDS = (401, 409, 419, 421)
TERMS = 8
REFINEMENT_CAP = 128
REFERENCE_CAP = 1024
REFERENCE_WIDTH = 1e-10
SMALL = 1e-7
REPETITIONS = 5
INPUT_FILES = [OLD / "code/intervals.py", OLD / "code/independent_audit.py", OLD / "inputs.json",
               OLD / "data/optdigits.tes"]
INPUT_FILES += [OLD / f"results/{seed}/{name}" for seed in SEEDS
                for name in ("weights.npz", "coarse.npz", "refined.npz", "test-classes.npz")]


def _legacy():
    modules = []
    for name, filename in (("stage5_legacy_intervals", "intervals.py"), ("stage5_independent_global", "independent_audit.py")):
        spec = importlib.util.spec_from_file_location(name, OLD / "code" / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def _empty(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Numerical-bound output folder is not empty; stop.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _exact(a, b, label):
    if np.asarray(a).dtype != np.asarray(b).dtype or not np.array_equal(a, b):
        raise RuntimeError("Exact numerical replay failed: " + label)


def _logical_hash(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        original = np.asarray(arrays[name])
        array = np.ascontiguousarray(original)
        digest.update(json.dumps([name, original.dtype.str, list(original.shape)], separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_arrays(arrays):
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError("Nonfinite numerical evidence: " + name)


def _save_reload(path, arrays):
    _finite_arrays(arrays)
    np.savez_compressed(path, **arrays)
    saved = _load_npz(path)
    if set(saved) != set(arrays):
        raise RuntimeError("Saved numerical array list differs; stop.")
    for name in arrays:
        _exact(saved[name], arrays[name], "saved " + name)
    return saved


def _peak():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def _journal(journal, phase, step, arrays, history=None):
    if journal is None:
        return
    snapshot = {name: np.asarray(value).copy() for name, value in arrays.items()}
    record = dict(phase=phase, attempted_step=int(step), arrays=snapshot,
                  completed_history={} if history is None else {name: tuple(parts) for name, parts in history.items()},
                  context=journal[0].get("context", {}))
    # One assignment publishes a complete snapshot. Interrupted copies leave
    # the previous complete snapshot in place.
    journal[0] = record


def _failure(out, stem, journal, error):
    # Model work has stopped. Disable the repeating guard alarm only for this
    # bounded final journal write; the caller immediately re-raises afterward.
    # Otherwise an expired permit can interrupt its own failure evidence.
    signal.setitimer(signal.ITIMER_REAL, 0)
    record = journal[0]
    arrays = dict(record.get("context", {}))
    arrays.update(record.get("arrays", {}))
    for name, parts in record.get("completed_history", {}).items():
        if parts:
            arrays["completed_history_" + name] = np.concatenate(parts)
    arrays.update(record.get("offending", {}))
    total_bytes = sum(value.nbytes for value in arrays.values())
    # The fixed 32-state/3,840-pair/129-row histories fit this bound. Never run
    # a new model or a replay during failure saving.
    if total_bytes > 64 * 1024 ** 2:
        raise RuntimeError("Failure journal exceeded its fixed 64 MiB bound") from error
    np.savez_compressed(out / (stem + "-partial.npz"), **arrays)
    _json(out / (stem + "-failure.json"), dict(status="stopped", phase=record.get("phase", "loading"),
          attempted_step=record.get("attempted_step", 0), known_complete_state_boundary=True,
          error_type=type(error).__name__, reason=str(error), raw_array_bytes=total_bytes,
          automatic_resume_permitted=False))


def _map(state, w, injection, removed):
    future = np.maximum(state @ w.T + injection, 0.)
    rows = np.flatnonzero(removed >= 0)
    future[rows, removed[rows]] = 0.
    if not np.isfinite(future).all():
        raise RuntimeError("Nonfinite ordinary state update; stop.")
    return future


def _solve(w, injection, removed, guard, journal=None):
    state = np.zeros_like(injection)
    residual = np.zeros(len(state))
    residual_checked = np.zeros(len(state), bool)
    updates = np.zeros(len(state), np.int64)
    live = np.arange(len(state))
    calls = examples = 0
    for step in range(513):
        _journal(journal, "coarse", step, dict(state=state, completed_state_updates=updates,
                 active_rows=live, last_observed_residual=residual, residual_was_observed=residual_checked,
                 removed_coordinates=removed))
        guard()
        future = _map(state[live], w, injection[live], removed[live])
        calls += 1
        examples += len(live)
        difference = np.max(np.abs(future - state[live]), axis=1)
        residual[live] = difference
        residual_checked[live] = True
        keep = difference > .001
        if not np.any(keep):
            _journal(journal, "coarse", step, dict(state=state, completed_state_updates=updates,
                     active_rows=live, last_observed_residual=residual, residual_was_observed=residual_checked,
                     removed_coordinates=removed))
            return state, residual, updates, calls, examples
        if step == 512:
            raise RuntimeError("Coarse state solve reached its fixed cap; stop.")
        _journal(journal, "coarse", step, dict(state=state, completed_state_updates=updates,
                 active_rows=live, last_observed_residual=residual, residual_was_observed=residual_checked,
                 removed_coordinates=removed))
        state[live[keep]] = future[keep]
        updates[live[keep]] += 1
        live = live[keep]
    raise AssertionError("Unreachable coarse solve branch")


def _series(w, readout, winner, runner, iv):
    """Cache one outward series per distinct fixed class pair."""
    pairs, inverse = np.unique(np.column_stack((winner, runner)), axis=0, return_inverse=True)
    coefficient = iv.sub(iv.point(readout[pairs[:, 0]]), iv.point(readout[pairs[:, 1]]))
    v = iv.abs_upper(coefficient)
    total = np.zeros_like(v)
    matrix = np.abs(w)
    for _ in range(TERMS):
        total = iv.up(total + v)
        future = np.zeros_like(v)
        for coordinate in range(len(w)):
            future = iv.up(future + iv.up(v[:, coordinate, None] * matrix[coordinate, None, :]))
        v = future
    tail = iv.nonnegative_sum_upper(v)
    return dict(class_pairs=pairs, inverse=inverse, unique_weights=total, unique_tail=tail,
                weights=total[inverse], tail=tail[inverse])


def _component_residual(state, w, injection, removed, iv):
    return iv.abs_upper(iv.sub(iv.map_interval(state, w, injection, removed), iv.point(state)))


def _series_radius(residual, weights, tail, q, iv):
    finite_sum = iv.nonnegative_sum_upper(iv.up(weights * residual))
    state_error = iv.positive_division_upper(np.max(residual, axis=1), iv.down(1. - q))
    return iv.up(finite_sum + iv.up(tail * state_error))


def _candidate_bound(base, edit, w, injection, removed, readout, winner, runner, q, cache, indices, iv):
    rb = _component_residual(base, w, injection, np.full(len(base), -1), iv)
    re = _component_residual(edit, w, injection, removed, iv)
    weights, tail = cache["weights"][indices], cache["tail"][indices]
    radius = iv.up(_series_radius(rb, weights, tail, q, iv) + _series_radius(re, weights, tail, q, iv))
    coefficient = iv.sub(iv.point(readout[winner]), iv.point(readout[runner]))
    center = iv.sum_interval(iv.mul(coefficient, iv.sub(iv.point(base), iv.point(edit))))
    lower, upper = iv.valid((iv.down(center[0] - radius), iv.up(center[1] + radius)))
    return dict(lower=lower, upper=upper, radius=radius, center_lower=center[0], center_upper=center[1],
                residual_base_upper=np.max(rb, axis=1), residual_edit_upper=np.max(re, axis=1))


def _category(lower, upper, step):
    result = np.full(len(lower), "active", dtype="<U10")
    result[lower > 0.] = "positive"
    result[upper < 0.] = "negative"
    small = (result == "active") & (lower >= -SMALL) & (upper <= SMALL)
    result[small] = "small"
    if step == REFINEMENT_CAP:
        result[result == "active"] = "unresolved"
    return result


def _refine(arm, base, edit, w, point_injection, interval_injection, units, readout, winner, runner, q, cache, iv, guard, journal=None):
    base, edit = base.copy(), edit.copy()
    n = len(base)
    status = np.full(n, "active", dtype="<U10")
    steps = np.zeros(n, np.int64)
    names = ("lower", "upper", "radius", "center_lower", "center_upper", "residual_base_upper", "residual_edit_upper")
    final = {name: np.zeros(n) for name in names}
    history = {name: [] for name in ("pair", "step", *names)}
    calls = map_rows = cert_calls = cert_rows = 0
    previous_lo, previous_hi = np.full(n, -np.inf), np.full(n, np.inf)
    for step in range(REFINEMENT_CAP + 1):
        active = np.flatnonzero(status == "active")
        state_updates = steps.copy()
        state_updates[active] = step
        _journal(journal, "refinement", step, dict(base=base, edit=edit, status=status,
                 completed_paired_updates=state_updates, last_checked_steps=steps, active_rows=active,
                 **{"last_" + key: value for key, value in final.items()}), history)
        guard()
        if not len(active):
            break
        common = (base[active], edit[active], w, tuple(x[active] for x in interval_injection), units[active],
                  readout, winner[active], runner[active], q)
        values = iv.effect_interval(*common) if arm == "global" else _candidate_bound(*common, cache, active, iv)
        cert_calls += 1
        cert_rows += len(active)
        if np.any(np.maximum(previous_lo[active], values["lower"]) > np.minimum(previous_hi[active], values["upper"])):
            if journal is not None:
                journal[0] = dict(journal[0], offending=dict(offending_pair=active.copy(),
                    offending_lower=values["lower"].copy(), offending_upper=values["upper"].copy(),
                    previous_lower=previous_lo[active].copy(), previous_upper=previous_hi[active].copy()))
            raise RuntimeError("Successive validated intervals are disjoint; stop.")
        previous_lo[active], previous_hi[active] = values["lower"], values["upper"]
        history["pair"].append(active.astype(np.int32))
        history["step"].append(np.full(len(active), step, np.int16))
        for name in names:
            history[name].append(values[name].copy())
            final[name][active] = values[name]
        current = _category(values["lower"], values["upper"], step)
        status[active], steps[active] = current, step
        live = active[current == "active"]
        if len(live):
            _journal(journal, "refinement", step, dict(base=base, edit=edit, status=status,
                     completed_paired_updates=steps, last_checked_steps=steps, active_rows=live,
                     **{"last_" + key: value for key, value in final.items()}), history)
            base[live] = _map(base[live], w, point_injection[live], np.full(len(live), -1))
            edit[live] = _map(edit[live], w, point_injection[live], units[live])
            calls += 2
            map_rows += 2 * len(live)
            if np.any(edit[live, units[live]] != 0.):
                raise RuntimeError("Permanent coordinate removal failed; stop.")
    arrays = {"base": base, "edit": edit, "status": status, "extra_updates": steps, **final}
    arrays.update({"history_" + name: np.concatenate(value) for name, value in history.items()})
    _journal(journal, "refinement", int(steps.max()), dict(base=base, edit=edit, status=status,
             completed_paired_updates=steps, **{"last_" + key: value for key, value in final.items()}), history)
    counts = dict(refinement_map_calls=calls, refinement_state_rows=map_rows,
                  refinement_certificate_calls=cert_calls, refinement_certificate_rows=cert_rows,
                  refinement_extra_paired_updates=int(steps.sum()))
    return arrays, counts


def _reference(base, edit, w, point_injection, injection, units, readout, winner, runner, ref, guard, journal=None):
    """Separate original-global implementation. No candidate arrays are read."""
    base, edit = base.copy(), edit.copy()
    n = len(base)
    q = float(np.max(ref.upper_sum(np.abs(w))))
    if not q < 1.:
        raise RuntimeError("Independent contraction check failed; stop.")
    lower, upper = np.zeros(n), np.zeros(n)
    checked = np.zeros(n, bool)
    steps = np.zeros(n, np.int64)
    completed = np.zeros(n, bool)
    active = np.arange(n)
    solver = ref.Solver(w)
    calls = rows = 0
    for step in range(REFERENCE_CAP + 1):
        state_updates = steps.copy()
        state_updates[active] = step
        _journal(journal, "reference", step, dict(base=base, edit=edit, completed_paired_updates=state_updates,
                 last_checked_steps=steps, active_rows=active, last_lower=lower, last_upper=upper,
                 bound_was_checked=checked, width_completed=completed))
        guard()
        values = ref.effect_bounds(base[active], edit[active], w, tuple(x[active] for x in injection),
                                  units[active], readout, winner[active], runner[active], q)
        calls += 1
        rows += len(active)
        lower[active], upper[active] = values["lower"], values["upper"]
        checked[active] = True
        width = ref.upper(values["upper"] - values["lower"])
        done = width <= REFERENCE_WIDTH
        completed[active[done]] = True
        steps[active] = step
        if step == REFERENCE_CAP:
            break
        live = active[~done]
        if not len(live):
            break
        _journal(journal, "reference", step, dict(base=base, edit=edit, completed_paired_updates=steps,
                 last_checked_steps=steps, active_rows=live, last_lower=lower, last_upper=upper,
                 bound_was_checked=checked, width_completed=completed))
        base[live] = solver.advance(base[live], point_injection[live], np.full(len(live), -1))
        edit[live] = solver.advance(edit[live], point_injection[live], units[live])
        if np.any(edit[live, units[live]] != 0.):
            raise RuntimeError("Independent reference removal failed; stop.")
        active = live
    arrays = dict(reference_base=base, reference_edit=edit, reference_lower=lower, reference_upper=upper,
                  reference_extra_updates=steps, reference_width_pass=completed)
    _journal(journal, "reference", int(steps.max()), dict(base=base, edit=edit,
             completed_paired_updates=steps, last_lower=lower, last_upper=upper,
             bound_was_checked=checked, width_completed=completed))
    counts = dict(reference_map_calls=solver.calls, reference_state_rows=solver.examples,
                  reference_certificate_calls=calls, reference_certificate_rows=rows,
                  reference_extra_paired_updates=int(steps.sum()))
    return arrays, counts


def _acceptance(arrays, fatal=True):
    pair = arrays["history_pair"]
    lo, hi = arrays["history_lower"], arrays["history_upper"]
    ref_lo, ref_hi = arrays["reference_lower"][pair], arrays["reference_upper"][pair]
    disjoint = (lo > ref_hi) | (hi < ref_lo)
    opposite = ((lo > 0.) & (ref_hi < 0.)) | ((hi < 0.) & (ref_lo > 0.))
    contains = (lo <= ref_lo) & (hi >= ref_hi)
    width_ready = arrays["reference_width_pass"][pair]
    accepted_rows = contains & width_ready
    accepted_pairs = np.ones(len(arrays["status"]), bool)
    np.logical_and.at(accepted_pairs, pair, accepted_rows)
    report = dict(history_intervals=len(pair), containing_intervals=int(contains.sum()),
                  noncontaining_intervals=int((~contains).sum()), disjoint_intervals=int(disjoint.sum()),
                  opposite_sign_intervals=int(opposite.sum()), reference_width_failures=int((~arrays["reference_width_pass"]).sum()),
                  accepted_effects=int(accepted_pairs.sum()), effects=len(accepted_pairs),
                  reliability_gate_pass=bool(accepted_pairs.all()),
                  refinement_all_resolved=bool(np.all(arrays["status"] != "unresolved")),
                  reference_all_resolved=bool(arrays["reference_width_pass"].all()),
                  all_resolved=bool(np.all(arrays["status"] != "unresolved") and arrays["reference_width_pass"].all()))
    if fatal and (np.any(disjoint) or np.any(opposite)):
        raise RuntimeError("Reference contradicts a validated interval; stop.")
    return report


def _history_checks(arrays):
    pair, step = arrays["history_pair"], arrays["history_step"]
    steps = arrays["extra_updates"]
    counts = np.bincount(pair, minlength=len(steps))
    if not np.array_equal(counts, steps + 1) or np.any(step < 0) or np.any(step > steps[pair]):
        raise RuntimeError("Refinement history omits or adds an update; stop.")
    keys = pair.astype(np.int64) * (REFINEMENT_CAP + 1) + step
    if len(np.unique(keys)) != len(keys):
        raise RuntimeError("Duplicate refinement history row; stop.")
    lo, hi = arrays["history_lower"], arrays["history_upper"]
    stop = (lo > 0.) | (hi < 0.) | ((lo >= -SMALL) & (hi <= SMALL))
    final_row = step == steps[pair]
    if np.any(stop & ~final_row) or np.any(~stop & final_row & (step < REFINEMENT_CAP)):
        raise RuntimeError("Refinement does not stop at its first fixed decision; stop.")
    last = np.flatnonzero(final_row)
    order = np.argsort(pair[last])
    _exact(lo[last[order]], arrays["lower"], "history final lower")
    _exact(hi[last[order]], arrays["upper"], "history final upper")
    status = np.array(["positive" if a > 0. else "negative" if b < 0. else
                       "small" if a >= -SMALL and b <= SMALL else
                       "unresolved" if k == REFINEMENT_CAP else "active"
                       for a, b, k in zip(arrays["lower"], arrays["upper"], steps)], dtype="<U10")
    _exact(status, arrays["status"], "independent stop classification")


def _read_inputs(seed):
    weights = _load_npz(OLD / f"results/{seed}/weights.npz")
    coarse = _load_npz(OLD / f"results/{seed}/coarse.npz")
    prior = _load_npz(OLD / f"results/{seed}/refined.npz")
    classes = _load_npz(OLD / f"results/{seed}/test-classes.npz")
    ledger = json.loads((OLD / "inputs.json").read_text())
    panel = np.array([row["row"] for row in ledger["panel"]], dtype=np.int64)
    raw = np.loadtxt(OLD / "data/optdigits.tes", delimiter=",")
    x = raw[panel, :64] / 16.
    if len(panel) != 120 or len(coarse["pair_unit"]) != 3840:
        raise RuntimeError("The fixed numerical panel changed; stop.")
    _exact(panel, coarse["panel_rows"], "original panel")
    image, units = np.repeat(np.arange(120), 32), np.tile(np.arange(32), 120)
    _exact(image, coarse["pair_image"], "image-unit order")
    _exact(units, coarse["pair_unit"], "removed unit order")
    _exact(classes["winner"][panel][image], coarse["winner"], "fixed winner")
    _exact(classes["runner"][panel][image], coarse["runner"], "fixed runner")
    _finite_arrays(weights)
    return weights, coarse, prior, x


def _arm_inner(seed, arm, repetition, out, prior_artifact, iv, ref, guard, journal):
    started = time.monotonic()
    guard()
    weights, old, prior, x = _read_inputs(seed)
    w, b, matrix, readout = [weights[key] for key in ("recurrent.weight", "input.bias", "input.weight", "readout.weight")]
    q = iv.contraction_upper(w)
    point = x @ matrix.T + b
    injection = iv.affine_point(x, matrix, b)
    independent_injection = ref.affine_bounds(x, matrix, b)
    for actual, expected in zip(injection, independent_injection):
        _exact(actual, expected, "independent input enclosure")
    states, residuals, updates = [], [], []
    coarse_calls = coarse_rows = 0
    for removed in range(-1, 32):
        if states:
            journal[0]["context"] = dict(completed_coarse_states=np.stack(states),
                completed_coarse_residuals=np.stack(residuals), completed_coarse_updates=np.stack(updates))
        state, residual, count, calls, rows = _solve(w, point, np.full(120, removed), guard, journal)
        states.append(state); residuals.append(residual); updates.append(count)
        coarse_calls += calls; coarse_rows += rows
        if removed >= 0 and np.any(state[:, removed] != 0.):
            raise RuntimeError("Coarse removed coordinate is not zero; stop.")
    journal[0]["context"] = dict(completed_coarse_states=np.stack(states),
        completed_coarse_residuals=np.stack(residuals), completed_coarse_updates=np.stack(updates))
    repeated, rr, rc, calls, rows = _solve(w, point, np.full(120, -1), guard, journal)
    coarse_calls += calls; coarse_rows += rows
    _exact(repeated, states[0], "unchanged coarse solve")
    _exact(rr, residuals[0], "unchanged coarse residual")
    _exact(rc, updates[0], "unchanged coarse update count")
    states, residuals, updates = np.stack(states), np.stack(residuals), np.stack(updates)
    for name, value in (("states", states), ("residuals", residuals), ("updates", updates)):
        _exact(value, old[name], "saved coarse " + name)
    image, units, winner, runner = [old[name] for name in ("pair_image", "pair_unit", "winner", "runner")]
    base, edit = states[0, image], states[units + 1, image]
    cache = _series(w, readout, winner, runner, iv) if arm == "componentwise" else None
    pair_injection = tuple(value[image] for value in injection)
    refined, counters = _refine(arm, base, edit, w, point[image], pair_injection, units, readout, winner, runner, q, cache, iv, guard, journal)
    if arm == "global":
        for name in ("base", "edit", "status", "extra_updates", "lower", "upper", "radius", "center_lower", "center_upper", "residual_base_upper", "residual_edit_upper"):
            _exact(refined[name], prior[name], "original global refinement " + name)
    # Rebuild the reference from a separate coarse-state copy for every arm.
    journal[0]["context"] = dict(journal[0].get("context", {}), **{"completed_refinement_" + key: value for key, value in refined.items()})
    reference, reference_counts = _reference(base, edit, w, point[image], tuple(value[image] for value in independent_injection),
                                              units, readout, winner, runner, ref, guard, journal)
    arrays = dict(coarse_states=states, coarse_residuals=residuals, coarse_updates=updates,
                  panel_rows=old["panel_rows"], pair_image=image, pair_unit=units, winner=winner, runner=runner,
                  q_upper=np.array(q), **refined, **reference)
    if cache is not None:
        arrays.update(series_class_pairs=cache["class_pairs"], series_index=cache["inverse"],
                      series_weights=cache["unique_weights"], series_tail=cache["unique_tail"])
    path = out / f"seed-{seed}-{arm}-repeat-{repetition}.npz"
    saved = _save_reload(path, arrays)
    _history_checks(saved)
    # Validate the final stored reference directly with its separate code.
    check = ref.effect_bounds(saved["reference_base"], saved["reference_edit"], w, tuple(value[image] for value in independent_injection),
                              units, readout, winner, runner, float(np.max(ref.upper_sum(np.abs(w)))))
    _exact(check["lower"], saved["reference_lower"], "saved reference lower endpoint")
    _exact(check["upper"], saved["reference_upper"], "saved reference upper endpoint")
    ready = ref.upper(check["upper"] - check["lower"]) <= REFERENCE_WIDTH
    _exact(ready, saved["reference_width_pass"], "saved reference width decision")
    if np.any(saved["reference_extra_updates"] > REFERENCE_CAP) or np.any((~ready) & (saved["reference_extra_updates"] != REFERENCE_CAP)):
        raise RuntimeError("Saved reference stop count is invalid; stop.")
    acceptance = _acceptance(saved, fatal=False)
    logical_hash = _logical_hash(saved)
    repeat_matches = True
    retained = True
    if prior_artifact is not None:
        original = _load_npz(prior_artifact)
        repeat_matches = (set(original) == set(saved) and _logical_hash(original) == logical_hash
                          and all(original[name].dtype == saved[name].dtype and np.array_equal(original[name], saved[name]) for name in saved))
        if repeat_matches:
            path.unlink()
            retained = False
    guard()
    elapsed = time.monotonic() - started
    report = dict(seed=seed, arm=arm, repetition=repetition, artifact_work_seconds=elapsed,
                  logical_sha256=logical_hash, repeat_matches=repeat_matches, full_artifact_retained=retained,
                  artifact=path.name if retained else prior_artifact.name, peak_process_memory_bytes=_peak(),
                  coarse_map_calls=coarse_calls, coarse_state_rows=coarse_rows,
                  coarse_state_updates=int(updates.sum() + rc.sum()), input_affine_enclosure_calls=2,
                  input_affine_enclosure_rows=240, saved_reference_validation_calls=1,
                  saved_reference_validation_rows=len(units),
                  series_class_pairs=0 if cache is None else len(cache["class_pairs"]),
                  series_terms=0 if cache is None else TERMS, **counters, **reference_counts, **acceptance,
                  final_positive=int(np.sum(saved["status"] == "positive")), final_negative=int(np.sum(saved["status"] == "negative")),
                  final_small=int(np.sum(saved["status"] == "small")), final_unresolved=int(np.sum(saved["status"] == "unresolved")))
    _json(out / f"seed-{seed}-{arm}-repeat-{repetition}.json", report)
    _acceptance(saved, fatal=True)
    return report, path if retained else prior_artifact


def _arm(seed, arm, repetition, out, prior_artifact, iv, ref, guard):
    journal = [dict(phase="loading", attempted_step=0, arrays={}, context={}, completed_history={})]
    try:
        return _arm_inner(seed, arm, repetition, out, prior_artifact, iv, ref, guard, journal)
    except BaseException as error:
        _failure(out, f"seed-{seed}-{arm}-repeat-{repetition}", journal, error)
        raise


def _exact_target(lower, upper, target, label):
    if not Fraction.from_float(float(lower)) <= target <= Fraction.from_float(float(upper)):
        raise RuntimeError("Exact rational target is outside its interval: " + label)


def _dummy_controls(iv, ref, guard):
    # A nonsymmetric nilpotent matrix makes orientation errors visible.
    w = np.array([[0., .25], [0., 0.]])
    readout = np.array([[1., 0.], [0., 0.]])
    cache = _series(w, readout, np.array([0]), np.array([1]), iv)
    q = iv.contraction_upper(w)
    radius = _series_radius(np.array([[.125, .5]]), cache["weights"], cache["tail"], q, iv)[0]
    if not Fraction.from_float(float(radius)) >= Fraction(1, 4) or radius - .25 > 1e-12:
        raise RuntimeError("Triangular matrix orientation control failed; stop.")
    cases = [
        ("positive", np.array([[1.25, 1.]]), np.array([[1., 0.]]), np.array([[1., 1.]]), 1, np.array([1., 0.]), Fraction(1, 4)),
        ("negative", np.array([[1.25, 1.]]), np.array([[1., 0.]]), np.array([[1., 1.]]), 1, np.array([-1., 0.]), Fraction(-1, 4)),
        ("zero", np.array([[1.25, 1.]]), np.array([[0., 1.]]), np.array([[1., 1.]]), 0, np.array([0., 1.]), Fraction(0)),
        ("no_change", np.array([[1., 0.]]), np.array([[1., 0.]]), np.array([[1., -1.]]), 1, np.array([1., 0.]), Fraction(0)),
        ("small", np.zeros((1, 2)), np.zeros((1, 2)), np.array([[0., 2. ** -40]]), 1, np.array([1., 0.]), Fraction(1, 2 ** 42)),
    ]
    records = []
    for label, base, edit, point, removed, coefficient, exact in cases:
        guard()
        readout = np.vstack((coefficient, np.zeros(2)))
        winner, runner, units = np.array([0]), np.array([1]), np.array([removed])
        cache = _series(w, readout, winner, runner, iv)
        injection = iv.point(point)
        values = _candidate_bound(base, edit, w, injection, units, readout, winner, runner, q, cache, np.array([0]), iv)
        reference = ref.effect_bounds(base, edit, w, injection, units, readout, winner, runner, q)
        _exact_target(values["lower"][0], values["upper"][0], exact, label + " candidate")
        _exact_target(reference["lower"][0], reference["upper"][0], exact, label + " reference")
        classification = _category(values["lower"], values["upper"], 0)[0]
        expected = label if label in ("positive", "negative", "small") else "small"
        if classification != expected:
            raise RuntimeError("Known dummy stop outcome differs: " + label)
        records.append(dict(case=label, classification=str(classification), exact_target=str(exact)))
    # Exact target 2^20; neither cap can resolve the very slow map from zero.
    slow = np.array([[1. - 2. ** -20]])
    point = np.ones((1, 1))
    readout = np.array([[1.], [0.]])
    winner, runner, units = np.array([0]), np.array([1]), np.array([0])
    qslow = iv.contraction_upper(slow)
    cache = _series(slow, readout, winner, runner, iv)
    result, _ = _refine("componentwise", np.zeros((1, 1)), np.zeros((1, 1)), slow, point,
                        iv.point(point), units, readout, winner, runner, qslow, cache, iv, guard)
    reference, _ = _reference(np.zeros((1, 1)), np.zeros((1, 1)), slow, point, iv.point(point),
                              units, readout, winner, runner, ref, guard)
    if result["status"][0] != "unresolved" or result["extra_updates"][0] != 128 or reference["reference_width_pass"][0] or reference["reference_extra_updates"][0] != 1024:
        raise RuntimeError("Full-cap unresolved dummy control failed; stop.")
    _exact_target(result["lower"][0], result["upper"][0], Fraction(2 ** 20), "unresolved candidate")
    _exact_target(reference["reference_lower"][0], reference["reference_upper"][0], Fraction(2 ** 20), "unresolved reference")
    # Positive sign has priority even inside the small-magnitude band.
    if _category(np.array([1e-9]), np.array([2e-9]), 128)[0] != "positive":
        raise RuntimeError("Sign-first dummy control failed; stop.")
    zero_w = np.zeros((2, 2))
    zero_readout = np.array([[1., -1.], [0., 0.]])
    zero_point = np.array([[.5, .25]])
    zero_base, zero_edit = zero_point.copy(), np.array([[0., .25]])
    zero_units = np.array([0])
    zero_cache = _series(zero_w, zero_readout, winner, runner, iv)
    zero_q = iv.contraction_upper(zero_w)
    zero_values = _candidate_bound(zero_base, zero_edit, zero_w, iv.point(zero_point), zero_units,
                                  zero_readout, winner, runner, zero_q, zero_cache, np.array([0]), iv)
    zero_reference = ref.effect_bounds(zero_base, zero_edit, zero_w, iv.point(zero_point), zero_units,
                                      zero_readout, winner, runner, zero_q)
    _exact_target(zero_values["lower"][0], zero_values["upper"][0], Fraction(1, 2), "zero-weight candidate")
    _exact_target(zero_reference["lower"][0], zero_reference["upper"][0], Fraction(1, 2), "zero-weight reference")
    return dict(passed=True, triangular_radius=float(radius), exact_cases=records,
                unresolved_candidate_steps=128, unresolved_reference_steps=1024,
                sign_first_pass=True, zero_weight_case_pass=True)


def _measure(function, guard, repeats=3):
    samples = []
    for _ in range(repeats):
        guard()
        started = time.monotonic()
        function()
        samples.append(time.monotonic() - started)
    return max(samples)


def prepare(out, guard):
    """Only analytic toy cases and full-size random dummy arrays are used."""
    started = time.monotonic()
    out = _empty(out)
    iv, ref = _legacy()
    controls = _dummy_controls(iv, ref, guard)
    rng = np.random.Generator(np.random.PCG64(2026102650))
    n, dimension = 3840, 32
    w = rng.normal(size=(dimension, dimension))
    w *= .875 / np.sum(np.abs(w), axis=1, keepdims=True)
    q = iv.contraction_upper(w)
    readout = rng.normal(size=(10, dimension))
    class_pairs = np.array([(a, b) for a in range(10) for b in range(10) if a != b])
    pair_classes = class_pairs[np.arange(n) % len(class_pairs)]
    winner, runner = pair_classes[:, 0], pair_classes[:, 1]
    units = np.arange(n) % dimension
    base = rng.uniform(0., 1., (n, dimension))
    edit = rng.uniform(0., 1., (n, dimension))
    edit[np.arange(n), units] = 0.
    point = rng.normal(size=(n, dimension))
    injection = iv.point(point)
    cache = _series(w, readout, winner, runner, iv)
    guard_seconds = _measure(lambda: guard(), guard)
    benchmark_journal = [dict(context={})]
    coarse_snapshot = dict(state=base[:120], completed_state_updates=np.zeros(120, np.int64),
        active_rows=np.arange(120), last_observed_residual=np.zeros(120), residual_was_observed=np.zeros(120, bool),
        removed_coordinates=units[:120])
    large_snapshot = dict(base=base, edit=edit, status=np.full(n, "active", dtype="<U10"),
                          completed_paired_updates=np.zeros(n, np.int64), active_rows=np.arange(n))
    for index in range(12):
        large_snapshot[f"checked_vector_{index}"] = np.zeros(n)
    snapshot_history = {str(index): [np.zeros(n)] * 129 for index in range(9)}
    def snapshot():
        _journal(benchmark_journal, "dummy", 0, large_snapshot, snapshot_history)
    coarse_map = _measure(lambda: (_journal(benchmark_journal, "dummy", 0, coarse_snapshot),
                                  _map(base[:120], w, point[:120], units[:120]),
                                  _journal(benchmark_journal, "dummy", 0, coarse_snapshot)), guard)
    paired_map = _measure(lambda: (_map(base, w, point, np.full(n, -1)), _map(edit, w, point, units)), guard)
    global_bound = _measure(lambda: (snapshot(), iv.effect_interval(base, edit, w, injection, units, readout, winner, runner, q), snapshot()), guard)
    candidate_bound = _measure(lambda: (snapshot(), _candidate_bound(base, edit, w, injection, units, readout, winner, runner, q, cache, np.arange(n), iv), snapshot()), guard)
    reference_bound = _measure(lambda: (snapshot(), ref.effect_bounds(base, edit, w, injection, units, readout, winner, runner, q), snapshot()), guard)
    construction = _measure(lambda: _series(w, readout, winner, runner, iv), guard)
    x, matrix, bias = rng.normal(size=(120, 64)), rng.normal(size=(32, 64)), rng.normal(size=32)
    affine = _measure(lambda: (iv.affine_point(x, matrix, bias), ref.affine_bounds(x, matrix, bias)), guard)
    # Full-cap history size. Random values prevent an unrealistically cheap
    # all-zero compression benchmark. These are serialization dummy arrays.
    history_size = n * 129
    dummy = dict(status=np.full(n, "unresolved", dtype="<U10"),
                 history_pair=np.tile(np.arange(n, dtype=np.int32), 129),
                 history_step=np.repeat(np.arange(129, dtype=np.int16), n),
                 history_lower=-rng.uniform(.1, 1., history_size), history_upper=rng.uniform(.1, 1., history_size),
                 reference_lower=np.zeros(n), reference_upper=np.zeros(n), reference_width_pass=np.ones(n, bool))
    dummy["extra_updates"] = np.full(n, 128, np.int64)
    dummy["lower"] = dummy["history_lower"][-n:].copy()
    dummy["upper"] = dummy["history_upper"][-n:].copy()
    for name in ("history_radius", "history_center_lower", "history_center_upper", "history_residual_base_upper", "history_residual_edit_upper"):
        dummy[name] = rng.normal(size=history_size)
    for name in ("base", "edit", "reference_base", "reference_edit"):
        dummy[name] = rng.normal(size=(n, dimension))
    dummy["coarse_states"] = rng.normal(size=(33, 120, dimension))
    path = out / "dummy-archive.npz"
    saving_started = time.monotonic()
    saved = _save_reload(path, dummy)
    original = _load_npz(path)
    if _logical_hash(saved) != _logical_hash(original):
        raise RuntimeError("Dummy logical hash differs; stop.")
    for name in saved:
        _exact(saved[name], original[name], "dummy repeat " + name)
    _history_checks(saved)
    _acceptance(saved)
    path.unlink()
    archive_seconds = time.monotonic() - saving_started
    common = 34 * 513 * coarse_map + 128 * paired_map + 1025 * reference_bound + 1024 * paired_map
    guard_calls_per_arm = 34 * 513 + 129 + 1025 + 6
    common += affine + reference_bound + archive_seconds + guard_calls_per_arm * guard_seconds + 2.
    global_arm = common + 129 * global_bound
    candidate_arm = common + 129 * candidate_bound + construction
    projection = 4. * (20. * (global_arm + candidate_arm) + 60.)
    report = dict(passed=True, dummy_only=True, controls=controls, projected_seconds=projection,
                  projection="four times the full coarse/refinement/reference caps for 20 global and 20 candidate arms, plus fixed loading and summary allowances",
                  maximum_of_three_pass_seconds=dict(guard=guard_seconds, coarse_map=coarse_map, paired_map=paired_map,
                      global_bound=global_bound, candidate_bound=candidate_bound, reference_bound=reference_bound,
                      series_construction=construction, both_input_enclosures=affine),
                  full_history_archive_replay_seconds=archive_seconds, dummy_q_upper=q,
                  projected_global_arm_seconds=global_arm, projected_candidate_arm_seconds=candidate_arm,
                  loading_allowance_per_arm_seconds=2., summary_allowance_seconds=60.,
                  projected_explicit_guard_calls_per_arm=guard_calls_per_arm,
                  elapsed_seconds=time.monotonic() - started)
    _json(out / "bounds-preparation.json", report)
    guard()
    return report


def run(out, guard):
    """Parent must verify the public plan, source hashes, and preparation."""
    started = time.monotonic()
    out = _empty(out)
    iv, ref = _legacy()
    records, model_results = [], []
    source_hashes = {str(path.relative_to(OLD)): hashlib.sha256(path.read_bytes()).hexdigest() for path in INPUT_FILES}
    for model_index, seed in enumerate(SEEDS):
        first = {"global": None, "componentwise": None}
        for repetition in range(REPETITIONS):
            order = ("global", "componentwise") if (model_index + repetition) % 2 == 0 else ("componentwise", "global")
            for arm in order:
                guard()
                arm_started = time.monotonic()
                record, path = _arm(seed, arm, repetition, out, first[arm], iv, ref, guard)
                record["elapsed_seconds"] = time.monotonic() - arm_started
                records.append(record)
                if first[arm] is None:
                    first[arm] = path
        selected = [record for record in records if record["seed"] == seed]
        global_times = [record["elapsed_seconds"] for record in selected if record["arm"] == "global"]
        candidate_times = [record["elapsed_seconds"] for record in selected if record["arm"] == "componentwise"]
        median_global = float(np.median(global_times))
        median_candidate = float(np.median(candidate_times))
        ratio = median_candidate / median_global
        reliable = all(record["reliability_gate_pass"] for record in selected)
        resolved = all(record["all_resolved"] for record in selected)
        repeats = all(record["repeat_matches"] for record in selected)
        model_results.append(dict(seed=seed, global_median_seconds=median_global, componentwise_median_seconds=median_candidate,
                                  median_time_ratio=ratio, reduction_fraction=1. - ratio,
                                  time_gate_pass=ratio <= .75, reliability_gate_pass=reliable,
                                  all_resolved=resolved, repeat_gate_pass=repeats,
                                  practical_gate_pass=bool(ratio <= .75 and reliable and resolved and repeats)))
    for path in INPUT_FILES:
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path.relative_to(OLD))]:
            raise RuntimeError("Original numerical evidence changed during the run; stop.")
    with (out / "timing-and-checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    _json(out / "model-comparison.json", model_results)
    _json(out / "source-hashes.json", source_hashes)
    summary = dict(passed=True, status="complete", models=4, unique_effects=15360,
                   paired_repetitions_per_model=5, timed_arms=40, terms=TERMS,
                   reliability_gate_pass=all(row["reliability_gate_pass"] for row in model_results),
                   repeat_gate_pass=all(row["repeat_gate_pass"] for row in model_results),
                   all_resolved=all(row["all_resolved"] for row in model_results),
                   practical_gate_pass=all(row["practical_gate_pass"] for row in model_results),
                   elapsed_seconds=time.monotonic() - started,
                   claim="A fixed-panel verified cost comparison; not a new contraction theorem or a general model failure rate")
    _json(out / "summary.json", summary)
    guard()
    return summary

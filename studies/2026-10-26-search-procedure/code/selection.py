"""Fixed null-bank selection tests. No scientific work runs on import."""
from io import BytesIO
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

MODELS = tuple(range(8))
CANDIDATE_COUNTS = (8, 32, 128)
N_ROWS = 768
N_EXAMPLES = 64
N_TRIALS = 2000
BATCH = 32
ALPHA = .05
TOL = 1e-10
CENTER_TOL = 1e-12
NORMAL = NormalDist()
Z95 = NORMAL.inv_cdf(.95)
Z975 = NORMAL.inv_cdf(.975)
SUBSET_IDS = np.array(sorted(range(1, 255), key=lambda mask:
    (hashlib.sha256(f"search-procedure-v1:subset:{mask}".encode()).hexdigest(), mask)), dtype=np.int64)


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _empty_output(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Selection output folder is not empty; preserve it and stop.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _close(a, b, label, tol=TOL):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if not np.isfinite(error) or error > tol:
        raise RuntimeError(f"Selection check failed: {label}: {error}")
    return error


def _finite(*arrays):
    if not all(np.isfinite(x).all() for x in arrays):
        raise RuntimeError("Nonfinite selection value; stop.")


def _seeds(model, count):
    selection = 2026102600 + model * 10000 + count
    return selection, selection + 5000


def _indices(seed, rows, trials=N_TRIALS):
    return np.random.Generator(np.random.PCG64(seed)).integers(0, rows, size=(trials, N_EXAMPLES), dtype=np.int64).astype(np.uint16)


def _standardize(means, variance):
    score = np.zeros_like(means, dtype=np.float64)
    np.divide(means * np.sqrt(N_EXAMPLES), np.sqrt(variance), out=score, where=variance > 0.)
    return score


def _wilson_upper(rejections, trials=N_TRIALS):
    p = rejections / trials
    z2 = Z975 ** 2
    center = p + z2 / (2. * trials)
    radius = Z975 * np.sqrt(p * (1. - p) / trials + z2 / (4. * trials ** 2))
    return float((center + radius) / (1. + z2 / trials))


def _simulate(bank, model, count, guard, trials=N_TRIALS):
    """Each row draw is shared across candidates. The two draws are independent."""
    guard()
    selected_bank = bank[:, :count]
    variance = np.mean(selected_bank ** 2, axis=0)
    selection_seed, validation_seed = _seeds(model, count)
    selection_indices = _indices(selection_seed, len(bank), trials)
    validation_indices = _indices(validation_seed, len(bank), trials)
    means = np.empty((trials, count), np.float64)
    chosen = np.empty(trials, np.int64)
    selection_mean = np.empty(trials, np.float64)
    validation_mean = np.empty(trials, np.float64)
    selection_z = np.empty(trials, np.float64)
    validation_z = np.empty(trials, np.float64)
    for start in range(0, trials, BATCH):
        guard()
        end = min(start + BATCH, trials)
        batch_means = np.mean(selected_bank[selection_indices[start:end]], axis=1)
        score = _standardize(batch_means, variance[None, :])
        winner = np.argmax(score, axis=1)
        rows = np.arange(end - start)
        means[start:end] = batch_means
        chosen[start:end] = winner
        selection_mean[start:end] = batch_means[rows, winner]
        selection_z[start:end] = score[rows, winner]
        validation_mean[start:end] = np.mean(selected_bank[validation_indices[start:end], winner[:, None]], axis=1)
        validation_z[start:end] = _standardize(validation_mean[start:end], variance[winner])
    positive_variance = variance[chosen] > 0.
    bonferroni = NORMAL.inv_cdf(1. - ALPHA / count)
    result = dict(selection_indices=selection_indices, validation_indices=validation_indices,
                  candidate_subset_ids=SUBSET_IDS[:count].copy(), known_variance=variance,
                  all_selection_means=means, selected_column=chosen,
                  selected_subset_id=SUBSET_IDS[chosen].copy(),
                  selected_selection_mean=selection_mean, selected_validation_mean=validation_mean,
                  selected_selection_z=selection_z, selected_validation_z=validation_z,
                  reject_reused=(selection_z > Z95) & positive_variance,
                  reject_validation=(validation_z > Z95) & positive_variance,
                  reject_bonferroni=(selection_z > bonferroni) & positive_variance)
    _finite(*result.values())
    return result


def _replay(bank, model, count, saved, guard):
    """Replay by row frequencies rather than the original indexed mean."""
    guard()
    trials = len(saved["selected_column"])
    selection_seed, validation_seed = _seeds(model, count)
    if not np.array_equal(saved["selection_indices"], _indices(selection_seed, len(bank), trials)):
        raise RuntimeError("Saved selection draws do not match the fixed seed; stop.")
    if not np.array_equal(saved["validation_indices"], _indices(validation_seed, len(bank), trials)):
        raise RuntimeError("Saved validation draws do not match the fixed seed; stop.")
    if not np.array_equal(saved["candidate_subset_ids"], SUBSET_IDS[:count]):
        raise RuntimeError("Saved candidate order changed; stop.")
    selected_bank = bank[:, :count]
    # A separate sum checks the bank's known population variance.
    variance = np.sum(selected_bank * selected_bank, axis=0) / len(bank)
    error = _close(variance, saved["known_variance"], "known variance")
    threshold = NORMAL.inv_cdf(1. - ALPHA / count)
    for start in range(0, trials, BATCH):
        guard()
        end = min(start + BATCH, trials)
        selection_frequencies = np.array([np.bincount(row.astype(np.int64), minlength=len(bank))
                                          for row in saved["selection_indices"][start:end]], dtype=np.float64)
        validation_frequencies = np.array([np.bincount(row.astype(np.int64), minlength=len(bank))
                                           for row in saved["validation_indices"][start:end]], dtype=np.float64)
        independent_means = (selection_frequencies @ selected_bank) / N_EXAMPLES
        error = max(error, _close(independent_means, saved["all_selection_means"][start:end], "all selection means"))
        score = np.zeros_like(independent_means)
        for column in range(count):
            if variance[column] > 0.:
                score[:, column] = independent_means[:, column] / np.sqrt(variance[column] / N_EXAMPLES)
        # Replay the tie rule on the recorded means. Near-ties in an alternate
        # summation order must not change the predeclared recorded-data choice.
        recorded_score = np.zeros_like(score)
        for column in range(count):
            if variance[column] > 0.:
                recorded_score[:, column] = saved["all_selection_means"][start:end, column] * np.sqrt(N_EXAMPLES) / np.sqrt(variance[column])
        winner = np.array([max(range(count), key=lambda j: (row[j], -j)) for row in recorded_score])
        if not np.array_equal(winner, saved["selected_column"][start:end]):
            raise RuntimeError("Saved winner or fixed tie rule differs; stop.")
        if not np.array_equal(saved["selected_subset_id"][start:end], SUBSET_IDS[winner]):
            raise RuntimeError("Saved selected subset ID differs; stop.")
        rows = np.arange(end - start)
        selected_mean = independent_means[rows, winner]
        validation_means = np.sum(validation_frequencies * selected_bank[:, winner].T, axis=1) / N_EXAMPLES
        selected_score = np.zeros(end - start)
        validated_score = np.zeros(end - start)
        active = variance[winner] > 0.
        selected_score[active] = selected_mean[active] / np.sqrt(variance[winner[active]] / N_EXAMPLES)
        validated_score[active] = validation_means[active] / np.sqrt(variance[winner[active]] / N_EXAMPLES)
        for actual, name in ((selected_mean, "selected_selection_mean"), (validation_means, "selected_validation_mean"),
                             (selected_score, "selected_selection_z"), (validated_score, "selected_validation_z")):
            error = max(error, _close(actual, saved[name][start:end], name))
        checks = ((selected_score > Z95) & active, (validated_score > Z95) & active, (selected_score > threshold) & active)
        for actual, name in zip(checks, ("reject_reused", "reject_validation", "reject_bonferroni")):
            if not np.array_equal(actual, saved[name][start:end]):
                raise RuntimeError(f"Independent rejection check failed for {name}; stop.")
    return error


def _reload(path, intended):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(intended):
            raise RuntimeError("Saved trial archive has missing or extra arrays; stop.")
        saved = {key: archive[key].copy() for key in archive.files}
    for key, value in intended.items():
        if saved[key].dtype != value.dtype or not np.array_equal(saved[key], value):
            raise RuntimeError(f"Saved trial array changed: {key}; stop.")
    return saved


def prepare(out, guard):
    """Hand-made controls and a synthetic timing test only."""
    started = time.monotonic()
    out = _empty_output(out)
    guard()
    known_means = np.array([[1., 0., -1., 0.]])
    known_variance = np.array([1., 1., 1., 0.])
    control_error = _close(_standardize(known_means, known_variance), [[8., 0., -8., 0.]], "known standardized means")
    known_score = _standardize(known_means, known_variance)[0]
    for threshold in (Z95, NORMAL.inv_cdf(1. - ALPHA / 128)):
        if not np.array_equal((known_score > threshold) & (known_variance > 0.), [True, False, False, False]):
            raise RuntimeError("One-sided decision control failed; stop.")
    if int(np.argmax(_standardize(np.array([[.5, .5, 0.]]), np.ones(3)), axis=1)[0]) != 0:
        raise RuntimeError("First-column tie control failed; stop.")
    zero = _simulate(np.zeros((N_ROWS, 128)), 0, 128, guard, trials=2)
    if any(np.any(zero[name]) for name in ("reject_reused", "reject_validation", "reject_bonferroni")):
        raise RuntimeError("Zero-variance control rejected; stop.")
    control_error = max(control_error, _replay(np.zeros((N_ROWS, 128)), 0, 128, zero, guard))
    tail_errors = {str(count): abs(count * (1. - NORMAL.cdf(NORMAL.inv_cdf(1. - ALPHA / count))) - ALPHA)
                   for count in CANDIDATE_COUNTS}
    if max(tail_errors.values()) > 1e-12 or abs(1. - NORMAL.cdf(Z95) - ALPHA) > 1e-12:
        raise RuntimeError("Normal-law analytic control failed; stop.")
    control_error = max(control_error, _close(_wilson_upper(0), Z975 ** 2 / (N_TRIALS + Z975 ** 2), "zero-count Wilson bound"),
                        _close(_wilson_upper(N_TRIALS), 1., "all-count Wilson bound"))
    rng = np.random.default_rng(2026102627)
    dummy = rng.normal(size=(N_ROWS, 128)) + .5 * rng.normal(size=(N_ROWS, 1))
    dummy -= dummy.mean(axis=0)
    benchmark_start = time.monotonic()
    trial = _simulate(dummy, 0, 128, guard)
    buffer = BytesIO()
    np.savez_compressed(buffer, **trial)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        reopened = {key: archive[key].copy() for key in archive.files}
    for key in trial:
        if not np.array_equal(trial[key], reopened[key]):
            raise RuntimeError("Dummy archive replay failed; stop.")
    dummy_replay_error = _replay(dummy, 0, 128, reopened, guard)
    benchmark_seconds = time.monotonic() - benchmark_start
    report = dict(passed=True, status="pass", dummy_only=True, control_error=control_error,
                  normal_tail_errors=tail_errors, dummy_replay_error=dummy_replay_error,
                  benchmark_seconds=benchmark_seconds, projected_seconds=4. * (24. * benchmark_seconds + 30.),
                  projection_rule="4 times (24 largest-condition synthetic timings plus 30 seconds for bank loading and summaries)",
                  elapsed_seconds=time.monotonic() - started)
    _json(out / "selection-preparation.json", report)
    return report


def _load_bank(path):
    with np.load(path, allow_pickle=False) as archive:
        bank = {key: archive[key].copy() for key in archive.files}
    required = {"centered_effects", "effects", "mean", "variance", "subset_ids", "row_ids", "rows"}
    if not required.issubset(bank):
        raise RuntimeError("Development calibration bank is missing required arrays; stop.")
    if bank["effects"].shape != (N_ROWS, 254) or bank["centered_effects"].shape != (N_ROWS, 254):
        raise RuntimeError("Development calibration bank has the wrong shape; stop.")
    if bank["effects"].dtype != np.float64 or bank["centered_effects"].dtype != np.float64:
        raise RuntimeError("Development calibration bank must use float64; stop.")
    if not np.array_equal(bank["subset_ids"], SUBSET_IDS):
        raise RuntimeError("Development calibration candidate order changed; stop.")
    if bank["row_ids"].shape != (N_ROWS,) or len(np.unique(bank["row_ids"])) != N_ROWS or bank["rows"].shape != (N_ROWS, 10):
        raise RuntimeError("Development calibration row identity is invalid; stop.")
    if bank["row_ids"].dtype != np.int64 or bank["rows"].dtype != np.int64 or bank["subset_ids"].dtype != np.int64:
        raise RuntimeError("Development bank identities must use int64; stop.")
    if np.any(bank["rows"][:, 3] != 0) or not np.array_equal(bank["row_ids"], 192 * bank["rows"][:, 0] + bank["rows"][:, 1]):
        raise RuntimeError("Calibration contains a non-development row or a wrong ledger ID; stop.")
    if np.any((bank["rows"][:, 9] != 0) & (bank["rows"][:, 9] != 1)):
        raise RuntimeError("Calibration contains an unknown donor context; stop.")
    _finite(*bank.values())
    means = np.mean(bank["effects"], axis=0)
    centered = bank["effects"] - means
    centered -= np.mean(centered, axis=0)
    constant = np.all(bank["effects"] == bank["effects"][:1], axis=0)
    centered[:, constant] = 0.
    error = _close(bank["centered_effects"], centered, "bank centering", CENTER_TOL)
    if np.any(bank["centered_effects"][:, constant] != 0.):
        raise RuntimeError("A constant bank column is not exactly zero; stop.")
    error = max(error, _close(np.mean(bank["centered_effects"], axis=0), np.zeros(254), "bank zero means", CENTER_TOL))
    variance = np.mean(bank["centered_effects"] ** 2, axis=0)
    active = variance > 0.
    if np.any(np.abs(np.mean(bank["centered_effects"], axis=0)[active]) / np.sqrt(variance[active]) > CENTER_TOL):
        raise RuntimeError("Bank residual mean exceeds its fixed relative limit; stop.")
    error = max(error, _close(bank["mean"], means, "published bank means", CENTER_TOL))
    error = max(error, _close(bank["variance"], variance, "published bank variances", CENTER_TOL))
    return bank["centered_effects"], error


def run(out, guard, calibration_path):
    """Parent must first publish and verify the development bank and code lock."""
    started = time.monotonic()
    out = _empty_output(out)
    calibration_path = Path(calibration_path)
    guard()
    bank_hashes = {}
    for model in MODELS:
        path = calibration_path / f"bank-seed-{model}.npz"
        bank_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _json(out / "selection-bank-hashes.json", bank_hashes)
    rows, controls = [], []
    for model in MODELS:
        guard()
        path = calibration_path / f"bank-seed-{model}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != bank_hashes[path.name]:
            raise RuntimeError("Development calibration bank changed during the run; stop.")
        bank, centering_error = _load_bank(path)
        for count in CANDIDATE_COUNTS:
            guard()
            arrays = _simulate(bank, model, count, guard)
            saved_path = out / f"selection-seed-{model}-candidates-{count}.npz"
            np.savez_compressed(saved_path, **arrays)
            reopened = _reload(saved_path, arrays)
            replay_error = _replay(bank, model, count, reopened, guard)
            selection_seed, validation_seed = _seeds(model, count)
            controls.append(dict(model=model, candidates=count, passed=True, bank_centering_error=centering_error,
                                 saved_trial_replay_error=replay_error, selection_seed=selection_seed,
                                 validation_seed=validation_seed))
            for method, key in (("reused", "reject_reused"), ("independent_validation", "reject_validation"),
                                ("bonferroni", "reject_bonferroni")):
                rejected = int(np.sum(reopened[key]))
                upper = _wilson_upper(rejected)
                rows.append(dict(model=model, candidates=count, method=method, trials=N_TRIALS,
                                 rejected=rejected, rejection_rate=rejected / N_TRIALS,
                                 wilson_upper_95=upper, meets_7_percent_bound=upper <= .07,
                                 zero_variance_candidates=int(np.sum(reopened["known_variance"] == 0.))))
            guard()
    with (out / "selection-rates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [row for row in rows if row["method"] == "independent_validation"]
    gate = all(row["meets_7_percent_bound"] for row in primary)
    _json(out / "selection-controls.json", controls)
    summary = dict(passed=True, status="complete", measurement_controls_pass=True,
                   primary_calibration_gate_pass=gate, primary_conditions_passed=sum(row["meets_7_percent_bound"] for row in primary),
                   primary_conditions=24, trials_per_condition=N_TRIALS, total_trials=24 * N_TRIALS,
                   methods=3, threshold=Z95, alpha=ALPHA,
                   largest_primary_wilson_upper=max(row["wilson_upper_95"] for row in primary),
                   later_positive_claims_may_use_validation_rule=gate,
                   claim="Calibration of a normal approximation on centered development banks; not power or real-circuit error rates",
                   elapsed_seconds=time.monotonic() - started)
    _json(out / "selection-summary.json", summary)
    guard()
    return summary

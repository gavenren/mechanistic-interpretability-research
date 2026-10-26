"""Fixed sparse-code witness search for Experiment 2A. No work runs on import."""
from itertools import combinations
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "studies/2026-10-13-rare-feature-sae"
MODEL_HASHES = (
    "a5e8543a0e307d6e3c495d3426d8a3f9fa5f3c18bf769dac50d80fe0b29a97e8",
    "97244a51cc70ee6fd8d5feee023894dd119d6ef5f96f0e70627c3fbb76b55a97",
    "aecb70005a31abcc7d5ac7447999fb96c6114e054c09c24c77a95729066f8493",
    "0140de434008a0158172c8f8b7ff28517a77bf47a03bb98743947d68d7aada92",
    "c312c1b716d51f7ea82875dcb4d3d60138b553260dca949d6430142d17c48d6d",
    "104d08b8e6a6be9f711dd231cd2ff7811b7dcc6b4b756097f441df747d5af36e",
)
DICT_HASHES = (
    "f65a984477a473417b95affc20b02169701ad467e5ed3cf4685d5bac476fa68e",
    "051274e47977df7de76b9a7cd75ab759e3d653091af16e32d4a046a8cfe0dc35",
    "db770cd092ac5334dbf3ff7d69940209e357f304ed3261e17cbde2e7e829e989",
    "12f172d082ac7b8929340c07888517af55dfe1bb2ae1b0d2f8c158cfc7134487",
    "5b52348530476b078d6523701c68abff42d0451a7c625501c61f529ea118b783",
    "d758c6fad5f0177970267a05ab540d4be1f435db9015d5b6a152dd7421dac052",
)
CONTROL_TOL = 1e-9
NEG_TOL = 1e-10
ERROR_TOL = 1e-9
DISTINCT_TOL = 1e-7
RANK_REL_TOL = 1e-12
CHUNK = 256
INPUT_FILES = [str((SOURCE / "models" / f"seed-{s}_p-0.01_a-0.25_local.npz").relative_to(REPO)) for s in range(6)]
INPUT_FILES += [str((SOURCE / "data" / f"dictionary-seed-{s}.npz").relative_to(REPO)) for s in range(6)]
INPUT_FILES += [str((SOURCE / "data/generator-records.json").relative_to(REPO)),
                str((SOURCE / "code/run.py").relative_to(REPO))]


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _finite(*arrays):
    if not all(np.isfinite(a).all() for a in arrays):
        raise RuntimeError("Non-finite code value; stop.")


def _close(a, b, label):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if not np.isfinite(error) or error > CONTROL_TOL:
        raise RuntimeError(f"Code control failed: {label}: {error}")
    return error


def _encode(h, model):
    pre = (h - model["mean"]) @ model["Wenc"] + model["benc"]
    relu = np.maximum(pre, 0.)
    keep = np.argsort(-relu, axis=1, kind="stable")[:, :4]
    code = np.zeros_like(pre)
    np.put_along_axis(code, keep, np.take_along_axis(relu, keep, axis=1), axis=1)
    return code


def _encode_reference(h, model):
    rows = []
    for row in h:
        values = [max(0., sum(float((row[d] - model["mean"][d]) * model["Wenc"][d, j])
                             for d in range(len(row))) + float(model["benc"][j]))
                  for j in range(model["Wenc"].shape[1])]
        keep = sorted(range(len(values)), key=lambda j: (-values[j], j))[:4]
        rows.append([values[j] if j in keep else 0. for j in range(len(values))])
    return np.array(rows)


def _decode_reference(code, decoder, offset):
    return np.array([[sum(float(row[j] * decoder[j, d]) for j in range(len(row))) + float(offset[d])
                      for d in range(decoder.shape[1])] for row in code])


def _probability(h, u):
    logits = 8. * (h @ u / .25 - .5)
    return np.exp(-np.logaddexp(0., -logits))


def _replay_saved(path, expected, model):
    """Reopen every saved array, then check the stored edit evidence."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise RuntimeError("Saved code archive has missing or extra arrays; stop.")
        saved = {key: archive[key].copy() for key in archive.files}
    for key, intended in expected.items():
        if saved[key].dtype != intended.dtype or not np.array_equal(saved[key], intended):
            raise RuntimeError(f"Saved code array differs from intended array: {key}; stop.")
    _finite(*[value for value in saved.values() if np.issubdtype(value.dtype, np.number)])
    h, u = saved["h"], saved["rare_direction"]
    decoder = model["Wdec"]
    offset = model["mean"] + model["bdec"]
    error = _close(saved["baseline_code"], _encode_reference(h, model), "saved encoder replay")
    budget = saved["squared_error_budget"]
    for key in ("baseline", "low", "high", "witness", "same"):
        code = saved[f"{key}_code"]
        if np.any(code < 0.) or np.any(np.sum(code > 0., axis=1) > 4):
            raise RuntimeError("Saved-file code violates nonnegative TopK4 constraint; stop.")
        independent = _decode_reference(code, decoder, offset)
        error = max(error, _close(saved[f"{key}_reconstruction"], independent, f"saved {key} decode"))
        explicit_logit = [8. * (sum(float(row[d] * u[d]) for d in range(len(u))) / .25 - .5)
                          for row in independent]
        explicit_probability = np.array([1. / (1. + np.exp(-x)) if x >= 0. else np.exp(x) / (1. + np.exp(x))
                                         for x in explicit_logit])
        error = max(error, _close(saved[f"{key}_probability"], explicit_probability, f"saved {key} readout"))
        squared_error = np.array([sum(float((row[d] - h[i, d]) ** 2) for d in range(len(u)))
                                  for i, row in enumerate(independent)])
        if np.any(squared_error > budget + ERROR_TOL * np.maximum(1., budget)):
            raise RuntimeError("Saved-file code exceeds reconstruction budget; stop.")
        recorded_error = budget if key == "baseline" else saved[f"{key}_squared_error"]
        error = max(error, _close(recorded_error, squared_error, f"saved {key} error"))
    for key, mask_name in (("witness", "alternative_found"), ("same", "same_vector_found")):
        mask = saved[mask_name]
        distance = np.max(np.abs(saved[f"{key}_code"] - saved["baseline_code"]), axis=1)
        if np.any(distance[mask] <= DISTINCT_TOL) or np.any(distance[~mask] != 0.):
            raise RuntimeError("Saved-file witness flag is invalid; stop.")
    same_mask = saved["same_vector_found"]
    if np.any(same_mask):
        vector_distance = np.max(np.abs(saved["same_reconstruction"] - saved["baseline_reconstruction"]), axis=1)
        if np.any(vector_distance[same_mask] > CONTROL_TOL):
            raise RuntimeError("Saved same-vector witness changes decoded vector; stop.")
        probability_distance = np.abs(saved["same_probability"] - saved["baseline_probability"])
        if np.any(probability_distance[same_mask] > 8. * np.linalg.norm(u, 1) * CONTROL_TOL + CONTROL_TOL):
            raise RuntimeError("Saved same-vector witness violates readout bound; stop.")
    return error


def _search(decoder, y, original, u, k, guard):
    """Keep valid finite witnesses. The baseline is always in the output range.

    For each support: least-squares center plus two analytic extrema of the
    readout direction in its reconstruction-error ellipsoid. A candidate with
    a negative coefficient is rejected, apart from the fixed round-off rule.
    """
    n, width = original.shape
    decoded = original @ decoder
    budget = np.sum((decoded - y) ** 2, axis=1)
    tol = ERROR_TOL * np.maximum(1., budget)
    low = original.copy()
    high = original.copy()
    witness = original.copy()
    same = original.copy()
    low_value = decoded @ u
    high_value = low_value.copy()
    found = np.zeros(n, bool)
    same_found = np.zeros(n, bool)
    counts = np.zeros((n, k + 1, 3), np.int64)
    max_condition = 1.

    def consider(supports, coefficients, eligible, size, kind):
        nonlocal low_value, high_value
        _finite(coefficients)
        nonnegative = np.all(coefficients >= -NEG_TOL, axis=2)
        coefficients = np.maximum(coefficients, 0.)
        atoms = decoder[supports]
        recon = np.einsum("nmk,mkd->nmd", coefficients, atoms)
        errors = np.sum((recon - y[:, None, :]) ** 2, axis=2)
        _finite(recon, errors)
        valid = eligible & nonnegative & (errors <= budget[:, None] + tol[:, None])
        counts[:, size, kind] += np.sum(valid, axis=1)
        values = recon @ u
        for i in range(n):
            indices = np.flatnonzero(valid[i])
            if not len(indices):
                continue
            codes = np.zeros((len(indices), width))
            np.put_along_axis(codes, supports[indices], coefficients[i, indices], axis=1)
            differences = np.max(np.abs(codes - original[i]), axis=1)
            distinct = differences > DISTINCT_TOL
            if np.any(distinct) and not found[i]:
                witness[i] = codes[np.flatnonzero(distinct)[0]]
                found[i] = True
            same_vector = np.max(np.abs(recon[i, indices] - decoded[i]), axis=1) <= CONTROL_TOL
            if np.any(distinct & same_vector) and not same_found[i]:
                same[i] = codes[np.flatnonzero(distinct & same_vector)[0]]
                same_found[i] = True
            imin = int(np.argmin(values[i, indices]))
            imax = int(np.argmax(values[i, indices]))
            if values[i, indices[imin]] < low_value[i]:
                low_value[i] = values[i, indices[imin]]
                low[i] = codes[imin]
            if values[i, indices[imax]] > high_value[i]:
                high_value[i] = values[i, indices[imax]]
                high[i] = codes[imax]

    # The empty support is a separate, explicit candidate.
    consider(np.empty((1, 0), int), np.empty((n, 1, 0)), np.ones((n, 1), bool), 0, 0)
    for size in range(1, k + 1):
        all_supports = list(combinations(range(width), size))
        for start in range(0, len(all_supports), CHUNK):
            guard()
            supports = np.array(all_supports[start:start + CHUNK], dtype=int)
            atoms = decoder[supports]
            gram = np.einsum("mkd,mld->mkl", atoms, atoms)
            eigenvalues = np.linalg.eigvalsh(gram)
            if np.any(eigenvalues[:, 0] <= RANK_REL_TOL * eigenvalues[:, -1]):
                raise RuntimeError("Sparse support failed the fixed rank check; stop.")
            max_condition = max(max_condition, float(np.max(eigenvalues[:, -1] / eigenvalues[:, 0])))
            inverse = np.linalg.inv(gram)
            rhs = np.einsum("nd,mkd->nmk", y, atoms)
            center = np.einsum("mkl,nml->nmk", inverse, rhs)
            center_recon = np.einsum("nmk,mkd->nmd", center, atoms)
            center_error = np.sum((center_recon - y[:, None, :]) ** 2, axis=2)
            delta = budget[:, None] - center_error
            eligible = delta >= -tol[:, None]
            v = atoms @ u
            direction = np.einsum("mkl,ml->mk", inverse, v)
            variance = np.sum(v * direction, axis=1)
            if np.any(variance < -1e-20):
                raise RuntimeError("Negative ellipsoid variance; stop.")
            scale = np.zeros_like(delta)
            nonzero = variance > 1e-24
            scale[:, nonzero] = np.sqrt(np.maximum(delta[:, nonzero], 0.) / variance[None, nonzero])
            _finite(center, scale, direction)
            consider(supports, center, eligible, size, 0)
            consider(supports, center - scale[:, :, None] * direction[None, :, :], eligible, size, 1)
            consider(supports, center + scale[:, :, None] * direction[None, :, :], eligible, size, 2)
    guard()
    return dict(low=low, high=high, witness=witness, same=same, found=found,
                same_found=same_found, counts=counts, budget=budget,
                max_gram_condition=max_condition)


def prepare(out, guard):
    """Use hand-made arrays only. Do not load trained weights or study data."""
    start = time.monotonic()
    out = Path(out)
    guard()
    decoder = np.array([[1., 0.], [0., 1.], [1., 1.]])
    decoder[2] /= np.sqrt(2.)
    original = np.array([[1., 1., 0.]])
    result = _search(decoder, np.array([[1., 1.]]), original, np.array([1., 0.]), 2, guard)
    if not result["same_found"][0]:
        raise RuntimeError("Known duplicate-code control failed; stop.")
    same_error = _close(result["same"] @ decoder, [[1., 1.]], "known same-vector code")
    # The unit disk around [1,1] has x-coordinate range [0,2].
    bounded = _search(np.eye(2), np.array([[1., 1.]]), np.array([[1., 0.]]),
                      np.array([1., 0.]), 2, guard)
    disk_error = max(_close(bounded["low"] @ np.array([1., 0.]), [0.], "disk minimum"),
                     _close(bounded["high"] @ np.array([1., 0.]), [2.], "disk maximum"))
    # A negative coefficient is rejected. The finite search must retain valid outputs.
    for item in (result, bounded):
        for key in ("low", "high", "witness", "same"):
            if np.any(item[key] < 0.):
                raise RuntimeError("Nonnegative-code control failed; stop.")
    # A complete, same-size synthetic search gives the time estimate. No model
    # checkpoint, source input, or final case is opened during this test.
    guard()
    rng = np.random.default_rng(2626)
    benchmark_decoder = rng.normal(size=(24, 12))
    benchmark_decoder /= np.linalg.norm(benchmark_decoder, axis=1, keepdims=True)
    benchmark_original = np.zeros((48, 24))
    benchmark_original[:, :4] = rng.uniform(.2, 1., size=(48, 4))
    benchmark_y = benchmark_original @ benchmark_decoder + rng.normal(0., .1, size=(48, 12))
    benchmark_u = rng.normal(size=12)
    benchmark_u /= np.linalg.norm(benchmark_u)
    benchmark_start = time.monotonic()
    benchmark = _search(benchmark_decoder, benchmark_y, benchmark_original, benchmark_u, 4, guard)
    _finite(benchmark["low"], benchmark["high"])
    benchmark_seconds = time.monotonic() - benchmark_start
    report = dict(status="pass", passed=True, dummy_only=True, same_vector_error=same_error,
                  analytic_disk_error=disk_error, benchmark_seconds=benchmark_seconds,
                  projected_seconds=4. * (6. * benchmark_seconds + 30.),
                  projection_rule="4 times (6 full-size dummy search times plus 30 seconds for data, replay, files, and saved-file replay)",
                  elapsed_seconds=time.monotonic() - start)
    _json(out / "codes-preparation.json", report)
    return report


def _load_case(seed):
    model_path = SOURCE / "models" / f"seed-{seed}_p-0.01_a-0.25_local.npz"
    dictionary_path = SOURCE / "data" / f"dictionary-seed-{seed}.npz"
    records_path = SOURCE / "data/generator-records.json"
    if _hash(model_path) != MODEL_HASHES[seed] or _hash(dictionary_path) != DICT_HASHES[seed]:
        raise RuntimeError("Study 2 input hash changed; stop.")
    if _hash(records_path) != "71ef2dac1295669ba930e7e7fa30cde65bdcb32947f71d983dad20cfdbf3816b":
        raise RuntimeError("Study 2 generator record changed; stop.")
    with np.load(model_path) as saved:
        model = {name: saved[name].copy() for name in saved.files}
    with np.load(dictionary_path) as saved:
        u = saved["rare_direction"].copy()
        dictionary = saved["background_dictionary"].copy()
    record = json.loads(records_path.read_text())[seed]
    rng = np.random.default_rng(2000 + seed)
    banks = {}
    for name, number in (("train", 10000), ("validation", 1024), ("test", 2048)):
        coefficients = (rng.random((number, 20)) < .1) * rng.uniform(.5, 1.5, (number, 20))
        banks[name] = coefficients @ dictionary
        if _array_hash(banks[name]) != record["arrays"][name]["sha256"]:
            raise RuntimeError(f"Original data replay failed for seed {seed} {name}; stop.")
    inputs, splits, ids, bits = [], [], [], []
    for split, count in (("validation", 8), ("test", 16)):
        for index in range(count):
            for bit in (0, 1):
                inputs.append(banks[split][index] + .25 * bit * u)
                splits.append("development" if split == "validation" else "test")
                ids.append(f"{split}:{index}")
                bits.append(bit)
    _finite(*model.values(), u, np.array(inputs))
    return model, u, np.array(inputs), splits, ids, bits


def run(out, guard):
    """Run only after the parent verifies a public plan and code lock."""
    start = time.monotonic()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Code output directory is not empty; preserve it and stop.")
    out.mkdir(parents=True, exist_ok=True)
    rows, controls = [], []
    for seed in range(6):
        guard()
        model, u, h, splits, ids, bits = _load_case(seed)
        baseline = _encode(h, model)
        encoder_error = _close(baseline, _encode_reference(h, model), "independent encoder")
        decoder = model["Wdec"]
        offset = model["mean"] + model["bdec"]
        native = baseline @ decoder + offset
        noop_error = _close(native, _decode_reference(baseline, decoder, offset), "no change")
        result = _search(decoder, h - offset, baseline, u, 4, guard)
        arrays = dict(h=h, baseline_code=baseline, rare_direction=u,
                      baseline_reconstruction=native, baseline_probability=_probability(native, u),
                      bit=np.array(bits), sample_id=np.array(ids), split=np.array(splits),
                      candidate_counts=result["counts"], alternative_found=result["found"],
                      same_vector_found=result["same_found"], squared_error_budget=result["budget"])
        replay_error = 0.
        for key in ("low", "high", "witness", "same"):
            code = result[key]
            reconstruction = code @ decoder + offset
            independent = _decode_reference(code, decoder, offset)
            replay_error = max(replay_error, _close(reconstruction, independent, f"{key} decode replay"))
            probability = _probability(reconstruction, u)
            explicit_logit = np.array([8. * (sum(float(row[d] * u[d]) for d in range(len(u))) / .25 - .5)
                                       for row in independent])
            independent_probability = np.array([1. / (1. + np.exp(-x)) if x >= 0. else np.exp(x) / (1. + np.exp(x))
                                                 for x in explicit_logit])
            replay_error = max(replay_error, _close(probability, independent_probability, f"{key} readout replay"))
            error = np.sum((reconstruction - h) ** 2, axis=1)
            if np.any(code < 0.) or np.any(np.sum(code > 0., axis=1) > 4):
                raise RuntimeError("Saved code violates nonnegative TopK4 constraint; stop.")
            if np.any(error > result["budget"] + ERROR_TOL * np.maximum(1., result["budget"])):
                raise RuntimeError("Saved code exceeds reconstruction budget; stop.")
            arrays[f"{key}_code"] = code
            arrays[f"{key}_reconstruction"] = reconstruction
            arrays[f"{key}_probability"] = probability
            arrays[f"{key}_squared_error"] = error
        same_mask = result["same_found"]
        same_probability_error = 0.
        if np.any(same_mask):
            # Same-vector tolerance can amplify by the fixed readout slope.
            # Replay itself remains under CONTROL_TOL; report this difference.
            same_probability_error = float(np.max(np.abs(arrays["same_probability"][same_mask] - arrays["baseline_probability"][same_mask])))
            if same_probability_error > 8. * np.linalg.norm(u, 1) * CONTROL_TOL + CONTROL_TOL:
                raise RuntimeError("Same-vector readout bound failed; stop.")
        _finite(*[v for v in arrays.values() if np.issubdtype(v.dtype, np.number)])
        saved_path = out / f"codes-seed-{seed}.npz"
        np.savez_compressed(saved_path, **arrays)
        guard()
        saved_replay_error = _replay_saved(saved_path, arrays, model)
        control = dict(seed=seed, encoder_replay_error=encoder_error, no_change_error=noop_error,
                       decoded_replay_error=replay_error, same_vector_probability_difference=same_probability_error,
                       saved_file_replay_error=saved_replay_error,
                       max_gram_condition=result["max_gram_condition"], status="pass")
        controls.append(control)
        for i in range(len(h)):
            rows.append(dict(seed=seed, split=splits[i], background_id=ids[i], bit=bits[i],
                             alternative_found=bool(result["found"][i]), same_vector_found=bool(same_mask[i]),
                             valid_candidates=int(np.sum(result["counts"][i])),
                             baseline_squared_error=float(result["budget"][i]),
                             low_squared_error=float(arrays["low_squared_error"][i]),
                             high_squared_error=float(arrays["high_squared_error"][i]),
                             baseline_probability=float(arrays["baseline_probability"][i]),
                             low_probability=float(arrays["low_probability"][i]),
                             high_probability=float(arrays["high_probability"][i]),
                             probability_span=float(arrays["high_probability"][i] - arrays["low_probability"][i])))
        guard()
    with (out / "codes-cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _json(out / "codes-controls.json", controls)
    by_split = {}
    for split in ("development", "test"):
        selected = [row for row in rows if row["split"] == split]
        by_split[split] = dict(cases=len(selected), alternatives_found=sum(row["alternative_found"] for row in selected),
                              same_vector_found=sum(row["same_vector_found"] for row in selected),
                              mean_probability_span=float(np.mean([row["probability_span"] for row in selected])),
                              maximum_probability_span=max(row["probability_span"] for row in selected))
    summary = dict(status="complete", passed=True, cases=len(rows), models=6, support_count=12951,
                   candidate_construction="center and two readout extrema per nonempty support; empty support once",
                   claim="Finite constructed-code witnesses; not encoder behavior or code uniqueness",
                   by_split=by_split, controls_pass=True, elapsed_seconds=time.monotonic() - start)
    _json(out / "codes-summary.json", summary)
    return summary

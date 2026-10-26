"""Experiment 2A: fixed normalization procedure test. No training or download.

The parent runner must check the public source lock and usage permit first.
It must pass a callable guard(tokens=0) before it calls prepare or run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox as neox


REPO = Path(__file__).resolve().parents[3]
MODEL_ID = "EleutherAI/pythia-14m"
REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
SOURCE = "studies/2026-10-13-patch-calibration-transfer/code/run.py"
SOURCE_COMMIT = "168d952dee2fbb5a6a73c4c9c01bd32cd8bc52c5"
HASHES = {
    "config.json": "f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17",
    "model.safetensors": "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1",
    "tokenizer.json": "870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2",
    "tokenizer_config.json": "eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809",
    "special_tokens_map.json": "10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c",
    "README.md": "d1f2cf1d5181daedeaa70208ddd5cc5251867bde9acf6db7bb45a2265e25e163",
}
FRAMES = (
    "The word is {w}. The word is",
    "The answer is {w}. The answer is",
    "The label is {w}. The label is",
    "The code is {w}. The code is",
)
WORDS = ("cat", "dog", "book", "chair", "red", "blue", "rain", "snow")
SITES = tuple((i, name) for i in range(4)
              for name in ("input_layernorm", "post_attention_layernorm"))
ALPHAS = (0.1, 0.5, 1.0)
MODES = ("recomputed", "frozen")
TOL = 1e-8
TOKEN_CAP = 1_000_000
PREP_FORWARDS = 4
SCIENCE_FORWARDS = 32 * (1 + 8 * 4 * (1 + 3))
PYTHON_VERSION = "3.12.14"
SOFTWARE_VERSIONS = {
    "torch": "2.8.0", "numpy": "2.3.5", "transformers": "4.57.6",
    "huggingface-hub": "0.36.2", "safetensors": "0.6.2", "tokenizers": "0.22.2",
}


def empty_output(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Normalization output directory is not empty")
    return out


def save_npz_checked(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise RuntimeError("Saved array names differ: " + str(path))
        for name, intended in arrays.items():
            actual = saved[name]
            if actual.dtype != intended.dtype or not np.array_equal(actual, intended):
                raise RuntimeError("Saved array differs: " + str(path) + ":" + name)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True,
                              allow_nan=False) + "\n")
    tmp.replace(path)


def model_dir():
    return Path(os.environ.get("MI_PYTHIA_MODEL_DIR",
                               str(REPO.parent / "mi-staging" / "pythia-14m")))


def checked_inputs():
    if platform.python_version() != PYTHON_VERSION:
        raise RuntimeError("Fixed Python version differs.")
    software = {name: version(name) for name in SOFTWARE_VERSIONS}
    if software != SOFTWARE_VERSIONS:
        raise RuntimeError("Fixed software versions differ: " + repr(software))
    directory = model_dir()
    actual = {}
    for name, wanted in HASHES.items():
        actual[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual[name] != wanted:
            raise RuntimeError("Fixed model file differs: " + name)
    config = json.loads((directory / "config.json").read_text())
    assert config["hidden_size"] == 128 and config["num_hidden_layers"] == 6
    assert config["use_parallel_residual"] and config["rope_scaling"] is None
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    prompts = []
    for frame_id, frame in enumerate(FRAMES):
        for word in WORDS:
            text = frame.format(w=word)
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert 1 <= len(ids) <= 128
            prompts.append(dict(prompt_id=len(prompts), frame=frame_id,
                                word=word, text=text, token_ids=ids))
    score_ids = []
    for word in WORDS:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        assert len(ids) == 1, (word, ids)
        score_ids.append(ids[0])
    assert len(set(score_ids)) == 8 and len(prompts) == 32
    return directory, dict(model_id=MODEL_ID, model_revision=REVISION,
        python_version=PYTHON_VERSION, software_versions=software,
        files_sha256=actual, prior_code=SOURCE, prior_code_commit=SOURCE_COMMIT,
        prompts=prompts, score_words=list(WORDS), score_ids=score_ids,
        sites=[dict(layer=i, module=n) for i, n in SITES],
        edit_sizes=list(ALPHAS), absolute_tolerance=TOL,
        predicted_scientific_forwards=SCIENCE_FORWARDS,
        predicted_scientific_tokens=129 * sum(len(p["token_ids"]) for p in prompts),
        dummy_tokens=PREP_FORWARDS * 128,
        maximum_tokens_including_dummy=SCIENCE_FORWARDS * 128 + PREP_FORWARDS * 128)


def double_attention(module, query, key, value, attention_mask, scaling,
                     dropout=0.0, head_mask=None, **kwargs):
    """Same declared double arithmetic as Study 4; rotary tables stay shared."""
    assert query.dtype == key.dtype == value.dtype == torch.float64
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, :key.shape[-2]]
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=query.dtype)
    if head_mask is not None:
        weights = weights * head_mask
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


def load_model(directory):
    assert torch.get_num_threads() <= 2, "Parent runner must limit CPU threads."
    assert version("transformers") == "4.57.6"
    neox.eager_attention_forward = double_attention
    model = GPTNeoXForCausalLM.from_pretrained(directory, local_files_only=True,
        dtype=torch.float64, attn_implementation="eager").eval()
    model.requires_grad_(False)
    assert all(p.dtype == torch.float64 and p.device.type == "cpu"
               for p in model.parameters())
    return model


def manual_numpy(x, gamma, beta, eps, moments=None):
    """Independent formula; NumPy has no PyTorch LayerNorm call."""
    if moments is None:
        mean = np.mean(x, dtype=np.float64)
        variance = np.mean((x - mean) ** 2, dtype=np.float64)
    else:
        mean, variance = moments
    return (x - mean) / np.sqrt(variance + eps) * gamma + beta


def manual_torch(x, gamma, beta, eps, moments):
    mean, variance = moments
    return (x - mean) * torch.rsqrt(variance + eps) * gamma + beta


def assert_close(left, right, label):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Invalid arrays: " + label)
    error = float(np.max(np.abs(a - b)))
    if error > TOL:
        raise RuntimeError(f"{label}: {error:.17g} exceeds fixed {TOL:.17g}")
    return error


class Budget:
    """Local branch count plus parent permit, time, memory and token checks."""
    def __init__(self, guard):
        self.guard = guard
        self.tokens = 0
        self.forwards = 0

    def before(self, tokens):
        if self.tokens + tokens > TOKEN_CAP:
            raise RuntimeError("Normalization token cap reached.")
        self.guard(tokens=tokens)
        self.tokens += tokens
        self.forwards += 1


@torch.inference_mode()
def forward(model, ids, budget, capture=False, patch=None):
    """Each call charges the complete prompt, including control and replay."""
    budget.before(len(ids))
    handles, cache = [], {}
    if capture:
        for index, (layer, name) in enumerate(SITES):
            module = getattr(model.gpt_neox.layers[layer], name)

            def record_input(mod, args, index=index):
                cache[f"x_{index}"] = args[0][0, -1].cpu().numpy().copy()

            def record_output(mod, args, output, index=index):
                cache[f"y_{index}"] = output[0, -1].cpu().numpy().copy()

            handles.append(module.register_forward_pre_hook(record_input))
            handles.append(module.register_forward_hook(record_output))
    if patch is not None:
        site, baseline, alpha, mode, implementation = patch
        layer, name = SITES[site]
        module = getattr(model.gpt_neox.layers[layer], name)
        x = torch.from_numpy(baseline.copy())
        mean = x.mean()
        variance = ((x - mean) ** 2).mean()
        delta = torch.zeros_like(x)
        delta[0] = alpha * torch.sqrt(variance) * np.sqrt(x.numel())
        edited = x + delta
        gamma = module.weight.detach()
        beta = module.bias.detach()
        eps = float(module.eps)
        numpy_mean = np.mean(baseline, dtype=np.float64)
        numpy_variance = np.mean((baseline - numpy_mean) ** 2, dtype=np.float64)
        numpy_edited = baseline.copy()
        numpy_edited[0] += alpha * np.sqrt(numpy_variance) * np.sqrt(len(baseline))
        assert_close(edited.numpy(), numpy_edited, "independent hidden edit")

        def input_hook(mod, args):
            assert_close(args[0][0, -1].cpu().numpy(), baseline, "edit input")
            changed = args[0].clone()
            changed[0, -1] = edited
            return (changed,) + args[1:]

        def output_hook(mod, args, output):
            if implementation == "numpy":
                moments = (numpy_mean, numpy_variance) if mode == "frozen" else None
                vector = torch.from_numpy(manual_numpy(numpy_edited, gamma.numpy(),
                    beta.numpy(), eps, moments))
                changed = output.clone()
                changed[0, -1] = vector
            elif mode == "frozen":
                changed = output.clone()
                changed[0, -1] = manual_torch(edited, gamma, beta, eps, (mean, variance))
            else:
                changed = output
            cache["edited_input"] = edited.numpy().copy()
            cache["y"] = changed[0, -1].cpu().numpy().copy()
            return changed

        handles.append(module.register_forward_pre_hook(input_hook))
        handles.append(module.register_forward_hook(output_hook))
    try:
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
        scores = logits.cpu().numpy().copy()
        if not np.isfinite(scores).all():
            raise RuntimeError("Nonfinite model output.")
        return scores, cache
    finally:
        for handle in handles:
            handle.remove()


def prepare(out, guard):
    """Only artificial numerical checks and dummy model passes; no task output."""
    out = empty_output(out)
    started = time.monotonic()
    guard(tokens=0)
    directory, manifest = checked_inputs()
    write_json(out / "inputs.json", manifest)
    model = load_model(directory)
    # Artificial values test actual trained gain, bias, and epsilon.
    dummy = np.sin(np.arange(128, dtype=np.float64) * 0.37) + 0.2
    errors = []
    for index, (layer, name) in enumerate(SITES):
        guard(tokens=0)
        module = getattr(model.gpt_neox.layers[layer], name)
        gamma, beta = module.weight.detach().numpy(), module.bias.detach().numpy()
        for scale in (0.0, 0.25, 1.0, 4.0):
            x = dummy * scale
            tx = torch.from_numpy(x)
            with torch.inference_mode():
                native = module(tx).numpy()
                frozen = manual_torch(tx, module.weight, module.bias, module.eps,
                    (tx.mean(), ((tx - tx.mean()) ** 2).mean())).numpy()
            direct = manual_numpy(x, gamma, beta, module.eps)
            errors.append(dict(site=index, scale=scale,
                native_manual_error=assert_close(native, direct, "dummy native/manual"),
                frozen_no_change_error=assert_close(frozen, native, "dummy frozen/no change")))
    setup_seconds = time.monotonic() - started
    budget = Budget(guard)
    dummy_ids = [0, 1, 2, 3] * 32
    timings = []
    for i in range(PREP_FORWARDS):
        tick = time.monotonic()
        forward(model, dummy_ids, budget)
        elapsed = time.monotonic() - tick
        if i:
            timings.append(elapsed)
    # Fixed fourfold allowance; full 128-token pass used even for short prompts.
    projection = 4 * (max(timings) * SCIENCE_FORWARDS + setup_seconds + 30.0)
    status = dict(status="pass", passed=True, projected_seconds=projection,
        artificial_checks=errors, dummy_pass_seconds=timings,
        warmup_passes=1, timed_passes=3, dummy_tokens=budget.tokens,
        projected_scientific_seconds_fourfold=projection,
        measured_setup_seconds=setup_seconds, saving_allowance_seconds_before_factor=30.0,
        seconds=time.monotonic() - started, tolerance=TOL,
        machine=platform.machine(), processor=platform.processor(),
        software=manifest["software_versions"], python_version=PYTHON_VERSION,
        cpu_threads=torch.get_num_threads(), batch_size=1,
        precision="float64 weights and arithmetic; shared float32 rotary tables",
        task_outputs_used=False)
    write_json(out / "preparation.json", status)
    del model
    return status


def run(out, guard):
    """Run the locked panel. Save each finished prompt if a later stop occurs."""
    out = empty_output(out)
    started = time.monotonic()
    budget = Budget(guard)
    completed, aggregate = [], []
    arrays, rows, pid = {}, [], None
    try:
        guard(tokens=0)
        directory, manifest = checked_inputs()
        write_json(out / "inputs.json", manifest)
        model = load_model(directory)
        parameters = {}
        for index, (layer, name) in enumerate(SITES):
            module = getattr(model.gpt_neox.layers[layer], name)
            parameters[f"gamma_{index}"] = module.weight.detach().numpy().copy()
            parameters[f"beta_{index}"] = module.bias.detach().numpy().copy()
            parameters[f"eps_{index}"] = np.array(float(module.eps))
        save_npz_checked(out / "layernorm_parameters.npz", **parameters)
        score_ids = manifest["score_ids"]
        for prompt in manifest["prompts"]:
            guard(tokens=0)
            pid, ids = prompt["prompt_id"], prompt["token_ids"]
            base_scores, base = forward(model, ids, budget, capture=True)
            arrays = dict(baseline_logits=base_scores[score_ids], **base)
            rows = []
            for site in range(len(SITES)):
                baseline = base[f"x_{site}"]
                for alpha in (0.0,) + ALPHAS:
                    outputs, replays = {}, {}
                    for mode in MODES:
                        patch = (site, baseline, alpha, mode, "torch")
                        logits, cache = forward(model, ids, budget, patch=patch)
                        key = f"s{site}_a{alpha:g}_{mode}"
                        arrays[key + "_input"] = cache["edited_input"]
                        arrays[key + "_y"] = cache["y"]
                        arrays[key + "_logits"] = logits[score_ids]
                        replay_logits, replay = forward(model, ids, budget,
                            patch=(site, baseline, alpha, mode, "numpy"))
                        arrays[key + "_replay_y"] = replay["y"]
                        arrays[key + "_replay_logits"] = replay_logits[score_ids]
                        y_error = assert_close(cache["y"], replay["y"], "LayerNorm replay")
                        logit_error = assert_close(logits, replay_logits, "full-logit replay")
                        effect_error = assert_close(logits[score_ids] - base_scores[score_ids],
                            replay_logits[score_ids] - base_scores[score_ids], "score-effect replay")
                        ln_effect = float(np.linalg.norm(cache["y"] - base[f"y_{site}"]))
                        score_effect = float(np.max(np.abs(logits[score_ids] - base_scores[score_ids])))
                        assert_close(ln_effect, np.linalg.norm(replay["y"] - base[f"y_{site}"]),
                                     "LayerNorm effect length replay")
                        assert_close(score_effect,
                            np.max(np.abs(replay_logits[score_ids] - base_scores[score_ids])),
                            "score effect maximum replay")
                        control_error = 0.0
                        if alpha == 0:
                            control_error = max(
                                assert_close(cache["y"], base[f"y_{site}"], "zero-edit LayerNorm"),
                                assert_close(logits, base_scores, "zero-edit full logits"),
                                assert_close(replay_logits, base_scores, "zero-edit replay logits"))
                        rows.append(dict(prompt_id=pid, site=site, alpha=alpha, mode=mode,
                            is_no_change_control=(alpha == 0),
                            layernorm_replay_max_abs=y_error, full_logit_replay_max_abs=logit_error,
                            selected_effect_replay_max_abs=effect_error,
                            no_change_max_abs=control_error,
                            layernorm_effect_l2=ln_effect,
                            selected_score_effect_max_abs=score_effect))
                        outputs[mode] = (cache["y"], logits[score_ids])
                        replays[mode] = (replay["y"], replay_logits[score_ids])
                    ln_delta = outputs["recomputed"][0] - outputs["frozen"][0]
                    score_delta = outputs["recomputed"][1] - outputs["frozen"][1]
                    assert_close(ln_delta, replays["recomputed"][0] - replays["frozen"][0],
                                 "procedure difference replay")
                    assert_close(score_delta, replays["recomputed"][1] - replays["frozen"][1],
                                 "procedure score difference replay")
                    assert_close(np.linalg.norm(ln_delta),
                        np.linalg.norm(replays["recomputed"][0] - replays["frozen"][0]),
                        "procedure difference length replay")
                    assert_close(np.max(np.abs(ln_delta)),
                        np.max(np.abs(replays["recomputed"][0] - replays["frozen"][0])),
                        "procedure difference maximum replay")
                    assert_close(np.max(np.abs(score_delta)),
                        np.max(np.abs(replays["recomputed"][1] - replays["frozen"][1])),
                        "procedure score maximum replay")
                    aggregate.append(dict(prompt_id=pid, site=site, alpha=alpha,
                        layernorm_rule_gap_l2=float(np.linalg.norm(ln_delta)),
                        layernorm_rule_gap_max_abs=float(np.max(np.abs(ln_delta))),
                        selected_score_rule_gap_max_abs=float(np.max(np.abs(score_delta)))))
            save_npz_checked(out / f"prompt_{pid:02d}.npz", **arrays)
            write_json(out / f"prompt_{pid:02d}_checks.json", rows)
            completed.append(pid)
            write_json(out / "progress.json", dict(completed_prompt_ids=completed,
                forwards=budget.forwards, tokens=budget.tokens,
                seconds=time.monotonic() - started))
        assert budget.forwards == SCIENCE_FORWARDS
        assert budget.tokens == manifest["predicted_scientific_tokens"]
        write_json(out / "rule_gaps.json", aggregate)
        status = dict(status="pass", passed=True, completed_prompt_ids=completed,
            forwards=budget.forwards, tokens=budget.tokens, tolerance=TOL,
            seconds=time.monotonic() - started, result_scope="numerical procedure only",
            copying_or_semantic_ability_tested=False)
        write_json(out / "status.json", status)
        return status
    except BaseException as exc:
        if pid is not None and pid not in completed and arrays:
            np.savez_compressed(out / f"prompt_{pid:02d}_partial.npz", **arrays)
            write_json(out / f"prompt_{pid:02d}_partial_checks.json", rows)
        write_json(out / "rule_gaps_partial.json", aggregate)
        write_json(out / "status.json", dict(status="stopped", passed=False,
            reason=f"{type(exc).__name__}: {exc}", completed_prompt_ids=completed,
            forwards=budget.forwards, tokens=budget.tokens,
            seconds=time.monotonic() - started))
        raise

#!/usr/bin/env python3
"""Read-only audit. Does not import the study runner or fit a new estimator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
FRAMES = ['The word is {w}. The word is', 'The answer is {w}. The answer is',
          'The label is {w}. The label is', 'The code is {w}. The code is',
          'Remember this word: {w}. The word was', 'A card says {w}. The card says',
          'In this list, the item is {w}. The item is',
          'The name on the box is {w}. The name on the box is']
WORDS = [('cat', 'dog'), ('book', 'chair'), ('apple', 'bread'), ('rain', 'snow'),
         ('red', 'blue'), ('king', 'queen'), ('car', 'train'), ('hand', 'foot'),
         ('bird', 'fish'), ('pen', 'cup'), ('milk', 'rice'), ('sun', 'moon'),
         ('green', 'white'), ('boy', 'girl'), ('bus', 'boat'), ('head', 'face')]
SPLITS = ['calibration', 'new_words', 'new_frames', 'new_words_and_frames']
CONDITIONS = ['float32', 'double_shared_rotary']
CELL_NAMES = ['00', '10', '01', '11']
CHECKS = 0
NUMBERS = 0
ERRORS = {}


def check(ok, label):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise AssertionError(label)


def near(actual, expected, label, tolerance=1e-11, category='arithmetic'):
    global NUMBERS
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    check(a.shape == b.shape, label + ': shape')
    check(bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()), label + ': finite')
    delta = float(np.max(np.abs(a-b))) if a.size else 0.0
    ERRORS[category] = max(ERRORS.get(category, 0.0), delta)
    NUMBERS += a.size
    check(delta <= tolerance, f'{label}: difference {delta} exceeds {tolerance}')


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(x):
    a = np.asarray(x)
    return np.where(abs(a) <= 1e-6, 0, np.sign(a))


def validate_inputs(model_dir):
    from transformers import AutoTokenizer
    provenance = read_json(HERE/'results/provenance.json')
    check(provenance['revision'] == REVISION, 'model revision')
    check(provenance['model_id'] == 'EleutherAI/pythia-14m', 'model identifier')
    check(set(provenance['model_files']) == {'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md'}, 'complete model file receipt')
    for name, info in provenance['model_files'].items():
        path = model_dir/name
        check(digest(path) == info['sha256'], 'model hash: ' + name)
        check(path.stat().st_size == info['bytes'], 'model size: ' + name)
    for filename, key in [('analysis-plan.md', 'plan_sha256'), ('code/run.py', 'runner_sha256')]:
        path = HERE/filename
        check(digest(path) == provenance[key], 'recorded lock hash: ' + filename)
        remote = subprocess.check_output(['git', 'show', provenance['lock_commit'] + ':' + str(path.relative_to(REPO))], cwd=REPO)
        check(remote == path.read_bytes(), 'git lock content: ' + filename)
    manifest = read_json(HERE/'results/manifest.json')
    for filename, info in manifest.items():
        path = HERE/filename
        check(digest(path) == info['sha256'], 'result hash: ' + filename)
        check(path.stat().st_size == info['bytes'], 'result size: ' + filename)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompts = read_json(HERE/'results/prompts.json')
    cases = read_json(HERE/'results/directions.json')
    check(len(prompts) == len(cases) == 256, 'complete prompt and case sets')
    for pid, prompt in enumerate(prompts):
        frame, pair, side = pid//32, (pid//2)%16, pid%2
        split = SPLITS[2*(frame >= 4)+(pair >= 8)]
        expected_text = FRAMES[frame].format(w=WORDS[pair][side])
        check(prompt['prompt_id'] == pid and prompt['frame'] == frame and prompt['word_pair'] == pair and prompt['side'] == side, 'prompt identity')
        check(prompt['text'] == expected_text and prompt['word'] == WORDS[pair][side] and prompt['split'] == split, 'prompt design')
        expected_tokens = tokenizer.encode(expected_text, add_special_tokens=False)
        check(prompt['token_ids'] == expected_tokens, 'prompt tokenization')
        target = tokenizer.encode(' '+WORDS[pair][side], add_special_tokens=False)
        check(len(target) == 1 and prompt['target_id'] == target[0], 'single target token')
        case = cases[pid]
        expected_case = dict(case_id=pid, frame=frame, word_pair=pair, direction=side,
                             split=split, receiver_id=pid, donor_id=pid^1,
                             receiver_token=prompt['target_id'], donor_token=prompts[pid^1]['target_id'])
        check(case == expected_case, 'direction design')
        other = prompts[pid^1]['token_ids']
        check(len(other) == len(expected_tokens) and other[-1] == expected_tokens[-1], 'matched length and final token')
        check(sum(a != b for a,b in zip(other, expected_tokens)) == 1, 'one input token change')
    check(all(sum(c['split'] == split for c in cases) == 64 for split in SPLITS), 'split sizes')
    return provenance, prompts, cases, len(manifest)


def audit_saved(condition, prompts, cases):
    directory = HERE/'results'/condition
    z = np.load(directory/'activations.npz')
    primary = condition == 'float32'
    tolerance = 1e-4 if primary else 1e-9
    category = condition + '_norm'
    expected_shapes = {'branches': (256,6,2,128), 'skips': (256,6,128),
                       'blocks': (256,6,128), 'final_residuals': (256,4,128),
                       'target_logits': (256,6,4,2), 'gamma': (128,), 'beta': (128,),
                       'token_ids': (32,), 'unembedding_rows': (32,128)}
    for key, shape in expected_shapes.items():
        check(z[key].shape == shape, 'saved shape: '+key)
        check(bool(np.isfinite(z[key]).all()), 'finite saved array: '+key)
    check(z['branches'].dtype == (np.float32 if primary else np.float64), 'saved condition dtype')
    logits = z['target_logits'].astype(float)
    margins = logits[:,:,:,1]-logits[:,:,:,0]
    attention = margins[:,:,1]-margins[:,:,0]
    mlp = margins[:,:,2]-margins[:,:,0]
    joint = margins[:,:,3]-margins[:,:,0]
    interaction = margins[:,:,3]+margins[:,:,0]-margins[:,:,1]-margins[:,:,2]
    calibrate = np.array([c['split'] == 'calibration' for c in cases])
    correction = np.median(interaction[calibrate], axis=0)
    saved_correction = read_json(directory/'calibration-corrections.json')
    near([saved_correction['corrections'][str(l)] for l in range(6)], correction, 'calibration medians')
    check(saved_correction['estimator'] == 'median' and saved_correction['n_directed_cases_per_layer'] == 64, 'calibration metadata')
    predicted = attention+mlp
    corrected = predicted+correction
    error0, error1 = abs(joint-predicted), abs(joint-corrected)
    rows = read_csv(directory/'four-cell-results.csv')
    check(len(rows) == 1536, 'complete factorial rows')
    check(len({(int(r['case_id']), int(r['layer'])) for r in rows}) == 1536, 'unique factorial rows')
    for row in rows:
        cid, layer = int(row['case_id']), int(row['layer'])
        for key, value in cases[cid].items():
            check(str(value) == row[key], 'case metadata in table')
        check(row['condition'] == condition, 'condition in table')
        for cell, name in enumerate(CELL_NAMES):
            near([float(row['receiver_logit_'+name]), float(row['donor_logit_'+name]), float(row['margin_'+name])],
                 [*logits[cid,layer,cell], margins[cid,layer,cell]], 'raw cell table')
        names = ['attention_effect','mlp_effect','joint_effect','interaction','calibration_correction',
                 'uncorrected_prediction','corrected_prediction','uncorrected_absolute_error','corrected_absolute_error']
        expected = [attention[cid,layer],mlp[cid,layer],joint[cid,layer],interaction[cid,layer],correction[layer],
                    predicted[cid,layer],corrected[cid,layer],error0[cid,layer],error1[cid,layer]]
        near([float(row[n]) for n in names], expected, 'effect and error table')
    summaries = read_csv(directory/'summary.csv')
    check(len(summaries) == 54, 'summary group count')
    primary_result = None
    for row in summaries:
        cmask = np.array([c['split'] == row['split'] for c in cases])
        lmask = np.arange(6)>0
        if row['group_type'] == 'layer':
            lmask = np.arange(6) == int(row['group_id'])
        elif row['group_type'] == 'frame':
            cmask &= np.array([c['frame'] == int(row['group_id']) for c in cases])
        elif row['group_type'] == 'word_pair':
            cmask &= np.array([c['word_pair'] == int(row['group_id']) for c in cases])
        else:
            check(row['group_type'] == 'all', 'known summary type')
        indexes = np.ix_(cmask, lmask)
        e0, e1 = error0[indexes], error1[indexes]
        expected = dict(n_cells=e0.size, uncorrected_mae=e0.mean(), corrected_mae=e1.mean(),
                        mae_gain=(e0-e1).mean(), median_absolute_interaction=np.median(e0),
                        minimum_absolute_interaction=e0.min(), maximum_absolute_interaction=e0.max(),
                        improved_fraction=(e1<e0).mean(),
                        uncorrected_sign_error=(sign(predicted[indexes]) != sign(joint[indexes])).mean(),
                        corrected_sign_error=(sign(corrected[indexes]) != sign(joint[indexes])).mean())
        near([float(row[k]) for k in expected], list(expected.values()), 'summary table')
        if row['split'] == 'new_words_and_frames' and row['group_type'] == 'all':
            primary_result = {k: float(v) for k,v in expected.items()}
    check(primary_result is not None, 'primary summary exists')
    gain = (error0-error1).reshape(8,16,2,6)[4:,8:,:,1:]
    random = np.random.default_rng(20261013)
    draws = []
    for _ in range(10000):
        frames = random.choice(4, 4, replace=True)
        pairs = random.choice(8, 8, replace=True)
        draws.append(gain[frames][:,pairs].mean())
    bounds = np.percentile(draws, [2.5,97.5])
    bootstrap = read_json(directory/'bootstrap.json')
    near([bootstrap['low'],bootstrap['high']], bounds, 'crossed fixed-case reweighting')
    check(bootstrap['repetitions'] == 10000 and bootstrap['seed'] == 20261013, 'resampling settings')
    calculated_controls = {'max_layer0_mlp_difference': 0.0, 'max_layer0_interaction': float(abs(interaction[:,0]).max()),
                           'max_last_layer_reconstruction_error': 0.0,
                           'max_frozen_denominator_interaction': 0.0,
                           'max_last_layer_additive_residual_error': 0.0}
    tokens = list(z['token_ids'])
    gamma, beta, epsilon = z['gamma'].astype(float), z['beta'].astype(float), float(z['epsilon'])
    for case in cases:
        cid, rid, did = case['case_id'], case['receiver_id'], case['donor_id']
        mdiff = abs(z['branches'][rid,0,1].astype(float)-z['branches'][did,0,1].astype(float)).max()
        calculated_controls['max_layer0_mlp_difference'] = max(calculated_controls['max_layer0_mlp_difference'], float(mdiff))
        x = z['final_residuals'][cid].astype(float)
        # Project onto a single centered readout direction, unlike the runner's two-row product.
        readout = z['unembedding_rows'][tokens.index(case['donor_token'])].astype(float)-z['unembedding_rows'][tokens.index(case['receiver_token'])].astype(float)
        centered_readout = gamma*readout
        centered_readout -= centered_readout.mean()
        numerator = x@centered_readout
        scale = np.sqrt(x.var(axis=-1)+epsilon)
        normal = numerator/scale + beta@readout
        frozen = numerator/scale[0] + beta@readout
        near(normal, margins[cid,5], 'independent final norm', tolerance, category)
        near(frozen[3]+frozen[0], frozen[1]+frozen[2], 'frozen denominator affine identity', tolerance, category)
        near(x[0], z['blocks'][rid,5], 'intact final block', tolerance, category)
        for cell,(a,m) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
            expected_x = z['skips'][rid,5].astype(float)+z['branches'][did if a else rid,5,0].astype(float)+z['branches'][did if m else rid,5,1].astype(float)
            near(x[cell], expected_x, 'fixed receiver skip residual', tolerance, category)
        calculated_controls['max_last_layer_reconstruction_error'] = max(calculated_controls['max_last_layer_reconstruction_error'], float(abs(normal-margins[cid,5]).max()))
        calculated_controls['max_frozen_denominator_interaction'] = max(calculated_controls['max_frozen_denominator_interaction'], float(abs(frozen[3]+frozen[0]-frozen[1]-frozen[2])))
        calculated_controls['max_last_layer_additive_residual_error'] = max(calculated_controls['max_last_layer_additive_residual_error'], float(abs(x[3]+x[0]-x[1]-x[2]).max()))
    controls = read_json(directory/'controls.json')
    check(controls['passed'] is True and controls['absolute_tolerance'] == tolerance, 'declared controls passed')
    # Different arithmetic order can change the last few digits of the residual maximum.
    near([controls[k] for k in calculated_controls], list(calculated_controls.values()), 'control summary', 1e-10)
    for value in calculated_controls.values():
        check(value <= tolerance, 'independent numerical control')
    return z, interaction, primary_result, calculated_controls


def audit_model(condition, model_dir, prompts, cases, z):
    import torch
    from transformers import GPTNeoXForCausalLM
    from transformers.models.gpt_neox import modeling_gpt_neox as module
    stock_attention = module.eager_attention_forward
    def double_attention(layer, query, key, value, attention_mask, scaling, dropout=0.0, head_mask=None, **kwargs):
        check(query.dtype == key.dtype == value.dtype == torch.float64, 'double attention dtype')
        check(not layer.training and dropout == 0, 'attention evaluation mode')
        scores = (query@key.transpose(-1,-2))*scaling
        if attention_mask is not None:
            scores += attention_mask[...,:key.shape[-2]]
        probabilities = torch.softmax(scores, dim=-1)
        if head_mask is not None:
            probabilities *= head_mask
        return (probabilities@value).transpose(1,2).contiguous(), probabilities
    primary = condition == 'float32'
    dtype = torch.float32 if primary else torch.float64
    tolerance = 1e-4 if primary else 1e-9
    category = condition + '_model'
    if not primary:
        module.eager_attention_forward = double_attention
    model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=dtype,
                                              attn_implementation='eager').eval()
    model.requires_grad_(False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    check(model.config.use_parallel_residual and not model.training, 'model architecture and mode')
    check(all(p.dtype == dtype for p in model.parameters()), 'model parameter dtype')
    near(model.gpt_neox.final_layer_norm.weight.detach().numpy(), z['gamma'], 'norm weight')
    near(model.gpt_neox.final_layer_norm.bias.detach().numpy(), z['beta'], 'norm bias')
    near(model.gpt_neox.final_layer_norm.eps, z['epsilon'], 'norm epsilon')
    near(model.embed_out.weight.detach().numpy()[z['token_ids']], z['unembedding_rows'], 'unembedding rows')
    with torch.inference_mode():
        length = max(len(p['token_ids']) for p in prompts)
        rotary = model.gpt_neox.rotary_emb(torch.zeros(1,length,128,dtype=dtype),torch.arange(length)[None])
        rotary_hash = hashlib.sha256(b''.join(v.cpu().numpy().astype(np.float32).tobytes() for v in rotary)).hexdigest()
    behavior = {int(r['prompt_id']):r for r in read_csv(HERE/'results'/condition/'intact-behavior.csv')}
    check(len(behavior) == 256, 'complete intact behavior')
    base_logits = {}
    calls = 0
    @torch.inference_mode()
    def forward(pid, changed_layer=None, delta=None, capture=False):
        nonlocal calls
        handles, cache = [], {}
        if capture:
            def record(key, tuple_output=False):
                def hook(layer, inputs, output):
                    tensor = output[0] if tuple_output else output
                    cache[key] = tensor[0,-1].cpu().numpy().copy()
                return hook
            def record_input(key):
                def hook(layer, inputs):
                    cache[key] = inputs[0][0,-1].cpu().numpy().copy()
                return hook
            for l, block in enumerate(model.gpt_neox.layers):
                handles.extend([block.attention.register_forward_hook(record(('a',l), True)),
                                block.mlp.register_forward_hook(record(('m',l))),
                                block.register_forward_hook(record(('block',l), True)),
                                block.register_forward_pre_hook(record_input(('skip',l)))])
        if changed_layer is not None:
            def add_at_block(block, inputs, output):
                changed = output[0].clone()
                changed[0,-1] += torch.as_tensor(delta, dtype=dtype)
                return (changed,)+output[1:]
            handles.append(model.gpt_neox.layers[changed_layer].register_forward_hook(add_at_block))
        try:
            result = model(torch.tensor([prompts[pid]['token_ids']]),use_cache=False).logits[0,-1].cpu().numpy().copy()
            calls += 1
            return result, cache
        finally:
            for handle in handles:
                handle.remove()
    try:
        for p in prompts:
            pid = p['prompt_id']
            full, cache = forward(pid, capture=True)
            base_logits[pid] = full
            for l in range(6):
                for name,key in [('a','branches'),('m','branches'),('skip','skips'),('block','blocks')]:
                    expected = z[key][pid,l,0 if name == 'a' else 1] if key == 'branches' else z[key][pid,l]
                    near(cache[(name,l)], expected, 'cached activation replay', tolerance, category)
            row = behavior[pid]
            target, other = p['target_id'], prompts[pid^1]['target_id']
            near([float(row['target_logit']),float(row['alternative_logit'])],full[[target,other]], 'intact target logits', tolerance, category)
            check(row['pairwise_target_preferred'] == str(bool(full[target]>full[other])), 'intact pair preference')
            check(row['target_top1'] == str(bool(full.argmax()==target)) and int(row['top1_token_id']) == int(full.argmax()), 'intact full vocabulary choice')
            for key in ['split','frame','word_pair']:
                check(row[key] == str(p[key]), 'intact prompt metadata')
        for case in cases:
            cid, rid, did = case['case_id'], case['receiver_id'], case['donor_id']
            targets = [case['receiver_token'],case['donor_token']]
            for layer in range(6):
                near(base_logits[rid][targets], z['target_logits'][cid,layer,0], 'baseline factorial cell', tolerance, category)
                for cell, components in [(1,[0]),(2,[1]),(3,[0,1])]:
                    # Calculate in the model dtype, then add once at the whole-block output.
                    delta = (z['branches'][did,layer,components]-z['branches'][rid,layer,components]).sum(axis=0)
                    logits, _ = forward(rid, layer, delta)
                    near(logits[targets], z['target_logits'][cid,layer,cell], 'whole-block delta versus branch replacement', tolerance, category)
        return {'forwards': calls, 'absolute_tolerance':tolerance, 'shared_rotary_table_sha256':rotary_hash,
                'maximum_absolute_logit_or_activation_difference':ERRORS.get(category,0.0)}
    finally:
        module.eager_attention_forward = stock_attention
        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--tables-only', action='store_true', help='Skip model replay; this is not a full audit.')
    args = parser.parse_args()
    started = time.perf_counter()
    provenance, prompts, cases, count = validate_inputs(args.model_dir)
    results, interactions = {}, []
    for condition in CONDITIONS:
        z, interaction, summary, controls = audit_saved(condition,prompts,cases)
        replay = None if args.tables_only else audit_model(condition,args.model_dir,prompts,cases,z)
        results[condition] = dict(primary=summary,controls=controls,model_replay=replay)
        interactions.append(interaction)
        z.close()
    if not args.tables_only:
        check(results[CONDITIONS[0]]['model_replay']['shared_rotary_table_sha256'] == results[CONDITIONS[1]]['model_replay']['shared_rotary_table_sha256'], 'rotary tables shared across numerical conditions')
    difference = abs(interactions[1]-interactions[0])
    precision = read_json(HERE/'results/precision-comparison.json')
    near([precision['max_absolute_interaction_difference'],precision['median_absolute_interaction_difference']],
         [difference.max(),np.median(difference)], 'precision interaction differences')
    check(precision['interaction_sign_change_count'] == int((sign(interactions[0])!=sign(interactions[1])).sum()), 'precision sign changes')
    near(precision['primary_mae_gain_change'],results[CONDITIONS[1]]['primary']['mae_gain']-results[CONDITIONS[0]]['primary']['mae_gain'], 'precision primary score change')
    print(json.dumps(dict(status='passed',scope='tables only' if args.tables_only else 'full independent calculation and model replay',
                         lock_commit=provenance['lock_commit'], manifest_files_checked=count, checks=CHECKS,
                         numbers_compared=NUMBERS,maximum_absolute_differences=ERRORS,
                         conditions=results,elapsed_seconds=time.perf_counter()-started),indent=2))


if __name__ == '__main__':
    main()

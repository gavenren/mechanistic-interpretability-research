#!/usr/bin/env python3
"""Fixed local spike-message versus spike-node selection test."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (311, 313, 317, 331)
QUERIES = ('message', 'node')
TRACE = ('u1', 's1', 'm1', 'v1', 'u2', 's2', 'v2')
STEPS, SITE, WIDTH, BATCH, EPOCHS = 8, 3, 32, 128, 40
TOL, CAP = 1e-10, 1200
PACKAGES = {'numpy': '2.3.5', 'torch': '2.8.0'}
DATA_HASHES = {'optdigits.tra': 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
               'optdigits.tes': '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def finite(*values):
    for value in values:
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite tensor.')


def lock_check(commit):
    repo = ROOT.parents[1]
    found = subprocess.run(['git', 'rev-parse', commit+'^{commit}'], cwd=repo,
                           check=True, capture_output=True, text=True).stdout.strip()
    assert found == commit, 'Use a full immutable lock commit.'
    files = ['analysis-plan.md', 'code/run.py', 'inputs.json', 'sources.md', 'requirements.txt']
    files += ['data/'+name for name in DATA_HASHES]
    for name in files:
        path = ROOT/name
        rel = path.relative_to(repo).as_posix()
        recorded = subprocess.run(['git', 'show', commit+':'+rel], cwd=repo,
                                  check=True, capture_output=True).stdout
        assert recorded == path.read_bytes(), rel


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, overdrive):
        ctx.save_for_backward(overdrive)
        return (overdrive >= 0).to(overdrive.dtype)

    @staticmethod
    def backward(ctx, incoming):
        (overdrive,) = ctx.saved_tensors
        return incoming/(1+10*overdrive.abs()).square()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 32, dtype=torch.float64)
        self.layer2 = nn.Linear(32, 32, dtype=torch.float64)
        self.readout = nn.Linear(32, 10, dtype=torch.float64)

    def forward(self, x, query=None, mask=None, self_replace=False, leaf=False, trace=False):
        v1 = x.new_zeros((len(x), 32)); v2 = v1.clone()
        records = {key: [] for key in TRACE} if trace else None
        output_spikes = []; variable = None
        current = self.layer1(x)
        for t in range(STEPS):
            u1 = .9*v1+current
            s1 = Spike.apply(u1-1)
            message = s1
            if t == SITE and query is not None:
                if leaf:
                    variable = s1.detach().requires_grad_(True)
                    message = variable
                    s1 = variable if query == 'node' else s1.detach()
                else:
                    replacement = s1.clone() if self_replace else torch.zeros_like(s1)
                    message = torch.where(mask, replacement, s1)
                    if query == 'node':
                        s1 = message
            v1 = u1-s1
            u2 = .9*v2+self.layer2(message)
            s2 = Spike.apply(u2-1)
            v2 = u2-s2
            output_spikes.append(s2)
            if trace:
                for key, value in zip(TRACE, (u1, s1, message, v1, u2, s2, v2)):
                    records[key].append(value)
        logits = self.readout(torch.stack(output_spikes, dim=1).mean(dim=1))
        return logits, {k: torch.stack(v, dim=1) for k, v in records.items()} if trace else None, variable


def initialize(seed):
    torch.manual_seed(seed)
    model = Net()
    assert sum(p.numel() for p in model.parameters()) == 3466
    return model


def parameter_hash(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(name.encode()); result.update(parameter.detach().numpy().tobytes())
    return result.hexdigest()


def checkpoint(path, model):
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})


def tangent(model, baseline, winner, runner_up, query):
    """Forward 32-direction tangent calculation, without autograd."""
    n = len(winner)
    eye = torch.eye(32, dtype=torch.float64).expand(n, -1, -1)
    dv1 = -eye if query == 'node' else torch.zeros_like(eye)
    # Axis one is the perturbed first-layer event; axis two is the current unit.
    du2 = eye @ model.layer2.weight.detach().T
    slope = lambda u: (1+10*(u-1).abs()).square().reciprocal()
    ds2 = du2*slope(baseline['u2'][:, SITE])[:, None, :]
    dv2 = du2-ds2; total = ds2
    for t in range(SITE+1, STEPS):
        du1 = .9*dv1
        ds1 = du1*slope(baseline['u1'][:, t])[:, None, :]
        dv1 = du1-ds1
        du2 = .9*dv2+ds1 @ model.layer2.weight.detach().T
        ds2 = du2*slope(baseline['u2'][:, t])[:, None, :]
        dv2 = du2-ds2; total = total+ds2
    read = model.readout.weight.detach()[winner]-model.readout.weight.detach()[runner_up]
    return (total*read[:, None, :]).sum(-1)/STEPS


class Table:
    def __init__(self, path):
        self.file = Path(path).open('w', newline=''); self.writer = None; self.count = 0

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row)); self.writer.writeheader()
        self.writer.writerow(row); self.file.flush(); self.count += 1

    def close(self):
        self.file.close()


class Run:
    def __init__(self, out, dummy=False):
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic(); self.dummy = dummy
        self.calls = self.examples = self.updates = 0
        self.active_model = None; self.seed = -1
        self.latest = None
        self.tables = {name: Table(self.out/(name+'.csv')) for name in
                       ('training', 'controls', 'cases', 'events', 'regrets', 'joints')}
        self.maxima = {}

    def budget(self):
        if time.monotonic()-self.started > CAP:
            raise TimeoutError('Fixed 20-minute cap exceeded.')

    def forward(self, model, x, **kwargs):
        self.budget(); self.calls += 1; self.examples += len(x)
        result = model(x, **kwargs)
        self.latest = dict(input=x.detach(), logits=result[0].detach())
        if result[1] is not None:
            self.latest.update({k:v.detach() for k,v in result[1].items()})
        finite(result[0])
        if result[1] is not None:
            finite(*result[1].values())
        return result

    def check(self, condition, name, error, tol=0., cases=0):
        error = float(error.detach()) if isinstance(error,torch.Tensor) else float(error)
        ok = bool(np.isfinite(error) and error <= tol)
        self.tables['controls'].add(dict(seed=self.seed, condition=condition, name=name,
            error=error, tolerance=tol, cases=cases, passed=ok))
        self.maxima[name] = max(self.maxima.get(name, 0.), error)
        if not ok:
            raise AssertionError(f'{condition}/{name}: {error} > {tol}')

    def train(self, model, x, y, seed, epochs=EPOCHS):
        model.train(); self.active_model = model
        opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
            weight_decay=0, amsgrad=False, foreach=False, fused=False, maximize=False,
            capturable=False, differentiable=False)
        rng = np.random.default_rng(10000+seed); orders = []
        for epoch in range(epochs):
            order = rng.permutation(len(x)); orders.append(order)
            for start in range(0, len(x), BATCH):
                ids = order[start:start+BATCH]
                opt.zero_grad(set_to_none=True)
                logits, _, _ = self.forward(model, x[ids])
                loss = F.cross_entropy(logits, y[ids]); finite(loss)
                loss.backward(); finite(*(p.grad for p in model.parameters()))
                opt.step(); self.updates += 1; finite(*model.parameters())
                self.tables['training'].add(dict(seed=seed, epoch=epoch, batch=start//BATCH,
                    n=len(ids), loss=float(loss.detach()), global_update=self.updates))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return np.stack(orders)

    def close(self):
        for tab in self.tables.values():
            tab.close()


def trace_error(a, b, rows=None, before=None, fields=TRACE):
    maximum = 0.
    for key in fields:
        aa, bb = a[key], b[key]
        if before is not None:
            aa, bb = aa[:, :before], bb[:, :before]
        if rows is not None:
            aa, bb = aa[rows], bb[rows]
        if aa.numel():
            maximum = max(maximum, float((aa-bb).abs().max().detach()))
    return maximum


def edit_controls(run, name, query, mask, trace, base, logits, baseline):
    n = len(logits)
    run.check(name, 'prefix_exact', trace_error(trace, base, before=SITE), cases=n)
    run.check(name, 'target_message_zero', trace['m1'][:, SITE][mask].abs().max(), cases=n)
    keep = ~mask
    unchanged = max(float((trace['u1'][:, SITE]-base['u1'][:, SITE]).abs().max()),
        *(float((trace[key][:, SITE][keep]-base[key][:, SITE][keep]).abs().max())
          for key in ('s1', 'm1', 'v1')))
    run.check(name, 'untouched_site_values', unchanged, cases=n)
    inactive = ((base['s1'][:, SITE]*mask).sum(-1) == 0)
    inactive_error = trace_error(trace, base, rows=inactive)
    if inactive.any():
        inactive_error = max(inactive_error, float((logits[inactive]-baseline[inactive]).abs().max()))
    run.check(name, 'inactive_no_op', inactive_error, cases=int(inactive.sum()))
    if query == 'message':
        run.check(name, 'first_layer_trajectory_exact', trace_error(trace, base, fields=('u1','s1','v1')), cases=n)
    else:
        run.check(name, 'target_node_zero', trace['s1'][:, SITE][mask].abs().max(), cases=n)
        expected = base['v1'][:, SITE]+base['s1'][:, SITE]*mask
        run.check(name, 'reset_algebra', (trace['v1'][:, SITE]-expected).abs().max(), TOL, n)


def panel(run, model, x, y, ids, seed, save=True):
    n = len(x); run.seed = seed
    names = []; recorded = {key: [] for key in ('logits','s1','m1','s2')}; extras = {}
    partial = {}; partial_paths = []
    if save:
        for key in recorded:
            path=run.out/f'panel-{seed}-{key}.partial.npy';partial_paths.append(path)
            shape=(75,n,10) if key=='logits' else (75,n,8,32)
            arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64 if key=='logits' else np.uint8,shape=shape)
            arr[:]=np.nan if key=='logits' else 255;arr.flush();partial[key]=arr
    def retain(name, logits, trace, full=False):
        names.append(name); recorded['logits'].append(logits.detach().numpy())
        for key in ('s1','m1','s2'):
            recorded[key].append(trace[key].detach().numpy().astype(np.uint8))
        if full:
            for key in ('u1','u2','v1','v2'):
                extras[name+'_'+key] = trace[key].detach().numpy()
        if save:
            for key in recorded:
                partial[key][len(names)-1]=recorded[key][-1];partial[key].flush()
            dump(run.out/f'panel-{seed}-conditions.partial.json',names)
    with torch.no_grad():
        baseline, base, _ = run.forward(model, x, trace=True)
    retain('baseline', baseline, base, full=True)
    order = np.argsort(-baseline.numpy(), axis=-1, kind='stable')
    winner = torch.from_numpy(order[:,0].copy()); runner_up = torch.from_numpy(order[:,1].copy())
    indices = torch.arange(n)
    margin = baseline[indices, winner]-baseline[indices, runner_up]
    gradients = []
    for query in QUERIES:
        logits, trace, variable = run.forward(model, x, query=query, leaf=True, trace=True)
        derivative = torch.autograd.grad((logits[indices,winner]-logits[indices,runner_up]).sum(), variable)[0]
        finite(derivative)
        run.check('leaf_'+query, 'leaf_logits_exact', (logits-baseline).abs().max(), cases=n)
        for key in TRACE:
            run.check('leaf_'+query, 'leaf_'+key+'_exact', (trace[key]-base[key]).abs().max(), cases=n)
        with torch.no_grad():
            expected = tangent(model, base, winner, runner_up, query)
        run.check('leaf_'+query, 'tangent_derivative', (derivative-expected).abs().max(), TOL, n*32)
        gradients.append(derivative.numpy()); retain('leaf_'+query, logits, trace, full=True)
    gradients = np.stack(gradients)
    original_spike = base['s1'][:,SITE].numpy()
    predictions = gradients*original_spike[None]
    selected = np.argmax(predictions, axis=-1)
    top4 = np.argsort(-predictions, axis=-1, kind='stable')[:,:,:4]
    exact = np.empty((2,n,32)); joint_effects = []
    with torch.no_grad():
        for qi, query in enumerate(QUERIES):
            for unit in range(32):
                mask = torch.zeros(n,32,dtype=torch.bool); mask[:,unit] = True
                name = f'single_{query}_{unit}'
                logits, trace, _ = run.forward(model, x, query=query, mask=mask, trace=True)
                edit_controls(run,name,query,mask,trace,base,logits,baseline)
                exact[qi,:,unit] = (margin-(logits[indices,winner]-logits[indices,runner_up])).numpy()
                retain(name,logits,trace)
        rng = np.random.default_rng(20000+seed)
        random = np.stack([rng.choice(32,4,replace=False) for _ in range(n)])
        for qi,query in enumerate(QUERIES):
            for gi, selection_name in enumerate((*QUERIES,'random')):
                chosen = top4[gi] if gi<2 else random
                mask = torch.zeros(n,32,dtype=torch.bool)
                mask[indices[:,None],torch.from_numpy(chosen)] = True
                name = f'joint_{query}_{selection_name}'
                logits,trace,_ = run.forward(model,x,query=query,mask=mask,trace=True)
                edit_controls(run,name,query,mask,trace,base,logits,baseline)
                effects = (margin-(logits[indices,winner]-logits[indices,runner_up])).numpy()
                joint_effects.append(effects); retain(name,logits,trace)
                for row in range(n):
                    run.tables['joints'].add(dict(seed=seed,row=int(ids[row]),query=query,
                        selector=selection_name,units=' '.join(map(str,chosen[row])),effect=float(effects[row])))
        for query in QUERIES:
            mask = torch.ones(n,32,dtype=torch.bool)
            logits,trace,_ = run.forward(model,x,query=query,mask=mask,self_replace=True,trace=True)
            run.check('self_'+query,'self_logits_exact',(logits-baseline).abs().max(),cases=n)
            for key in TRACE:
                run.check('self_'+query,'self_'+key+'_exact',(trace[key]-base[key]).abs().max(),cases=n)
            retain('self_'+query,logits,trace)
        # Exact linear control: remove a final SECOND-layer spike only at readout.
        rates = base['s2'].mean(1)
        read_error = 0.
        for unit in range(32):
            changed = rates.clone(); amount = base['s2'][:,-1,unit]
            changed[:,unit] -= amount/STEPS
            logits = model.readout(changed)
            observed = margin-(logits[indices,winner]-logits[indices,runner_up])
            expected = amount*(model.readout.weight[winner,unit]-model.readout.weight[runner_up,unit])/STEPS
            read_error = max(read_error,float((observed-expected).abs().max()))
        run.check('readout','final_second_layer_linear',read_error,TOL,n*32)
    regrets = np.empty((n,2,2)); selected_effect = np.empty_like(regrets)
    for qi,query in enumerate(QUERIES):
        oracle = exact[qi].max(-1)
        for gi,graph in enumerate(QUERIES):
            selected_effect[:,qi,gi] = exact[qi,np.arange(n),selected[gi]]
            regrets[:,qi,gi] = oracle-selected_effect[:,qi,gi]
            for row in range(n):
                run.tables['regrets'].add(dict(seed=seed,row=int(ids[row]),query=query,graph=graph,
                    selected_unit=int(selected[gi,row]),exact_selected_effect=float(selected_effect[row,qi,gi]),
                    oracle_effect=float(oracle[row]),regret=float(regrets[row,qi,gi])))
    gain_message = selected_effect[:,0,0]-selected_effect[:,0,1]
    gain_node = selected_effect[:,1,1]-selected_effect[:,1,0]
    primary = .5*(regrets[:,1,0]+regrets[:,0,1]-regrets[:,1,1]-regrets[:,0,0])
    run.check('scores','primary_formula',np.max(abs(primary-.5*(gain_message+gain_node))),TOL,n)
    run.check('scores','regret_nonnegative',max(0.,float(-regrets.min())),TOL,n*4)
    correct = winner.numpy()==y.numpy()
    for row in range(n):
        run.tables['cases'].add(dict(seed=seed,row=int(ids[row]),label=int(y[row]),
            winner=int(winner[row]),runner_up=int(runner_up[row]),correct=bool(correct[row]),
            active_candidates=int(original_spike[row].sum()),
            any_positive_message=bool((exact[0,row]>0).any()),any_positive_node=bool((exact[1,row]>0).any()),
            zero_effects_message=int((exact[0,row]==0).sum()),zero_effects_node=int((exact[1,row]==0).sum()),
            selection_same=bool(selected[0,row]==selected[1,row]),gain_message=float(gain_message[row]),
            gain_node=float(gain_node[row]),primary=float(primary[row])))
        for qi,query in enumerate(QUERIES):
            for unit in range(32):
                run.tables['events'].add(dict(seed=seed,row=int(ids[row]),query=query,unit=unit,
                    active=int(original_spike[row,unit]),exact_effect=float(exact[qi,row,unit]),
                    predicted_message=float(predictions[0,row,unit]),predicted_node=float(predictions[1,row,unit])))
    sign = []
    for qi,query in enumerate(QUERIES):
        for gi,graph in enumerate(QUERIES):
            eligible = (original_spike!=0)&(predictions[gi]!=0)&(exact[qi]!=0)
            errors = eligible & (np.sign(predictions[gi])!=np.sign(exact[qi]))
            sign.append(dict(query=query,graph=graph,n=int(eligible.sum()),wrong=int(errors.sum()),
                             fraction=float(errors.sum()/eligible.sum()) if eligible.any() else None))
    result = dict(seed=seed,panel_n=n,panel_accuracy=float(correct.mean()),primary=float(primary.mean()),
        gain_message=float(gain_message.mean()),gain_node=float(gain_node.mean()),
        regret_table=regrets.mean(0).tolist(),selected_effect_table=selected_effect.mean(0).tolist(),
        active_candidates_mean=float(original_spike.sum(-1).mean()),
        any_positive_fraction=[float((exact[q]>0).any(-1).mean()) for q in range(2)],
        zero_effect_counts=[int((exact[q]==0).sum()) for q in range(2)],
        selection_agreement=float((selected[0]==selected[1]).mean()),wrong_sign=sign,
        correct_baseline_n=int(correct.sum()),correct_baseline_primary=float(primary[correct].mean()) if correct.any() else None,
        joint_means=np.stack(joint_effects).mean(-1).tolist(),
        exact_query_difference_mean=float((exact[1]-exact[0]).mean()),
        exact_query_difference_mean_absolute=float(abs(exact[1]-exact[0]).mean()))
    assert len(names)==75
    if save:
        arrays = {key:np.stack(value) for key,value in recorded.items()}
        arrays.update(extras); arrays.update(condition_names=np.array(names),row_indices=np.array(ids),labels=y.numpy(),
            gradients=gradients,predictions=predictions,exact_effects=exact,selected=selected,top4=top4,
            random_top4=random,regrets=regrets,primary=primary,gain_message=gain_message,gain_node=gain_node)
        np.savez_compressed(run.out/f'panel-{seed}.npz',**arrays)
        partial.clear()
        for path in partial_paths:path.unlink()
        (run.out/f'panel-{seed}-conditions.partial.json').unlink()
    return result


def data():
    ledger = json.loads((ROOT/'inputs.json').read_text())
    arrays=[]
    for name,count in [('optdigits.tra',3823),('optdigits.tes',1797)]:
        assert sha(ROOT/'data'/name)==DATA_HASHES[name]==ledger['data_hashes'][name]
        raw=np.loadtxt(ROOT/'data'/name,delimiter=',',dtype=np.int64)
        assert raw.shape==(count,65)
        assert ((raw[:,:64]>=0)&(raw[:,:64]<=16)).all() and ((raw[:,-1]>=0)&(raw[:,-1]<10)).all()
        arrays.append((torch.from_numpy(raw[:,:64]/16.),torch.from_numpy(raw[:,-1].copy())))
    labels=arrays[1][1].numpy()
    ids=sorted(np.concatenate([np.flatnonzero(labels==digit)[:12] for digit in range(10)]).tolist())
    assert ledger['panel']==[dict(row=i,label=int(labels[i])) for i in ids]
    return arrays,np.array(ids)


def final_summary(summaries):
    values=np.array([r['primary'] for r in summaries])
    rng=np.random.default_rng(2026101501)
    boots=values[rng.integers(0,4,(10000,4))].mean(1)
    return dict(primary='off_diagonal_minus_diagonal_single_event_regret',seed_values=values.tolist(),
        mean=float(values.mean()),percentile_95_interval=np.quantile(boots,[.025,.975],method='linear').tolist(),
        bootstrap_seed=2026101501,bootstrap_repeats=10000,directional_hypothesis_mean_positive=bool(values.mean()>0),
        ability_gate_all_seeds=all(r['ability_gate'] for r in summaries),
        limit='Four fixed seeds and a small file-order explanation panel. The directional rule is not a significance or practical-success claim.')


def nonvacuous_dummy(run):
    model=initialize(99001)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.layer1.bias[0]=.7
        model.layer2.weight[0,0]=1.1
        model.readout.weight[0,0]=1.
    for p in model.parameters(): p.requires_grad_(False)
    x=torch.zeros(2,64,dtype=torch.float64)
    with torch.no_grad():
        z,b,_=model(x,trace=True)
        mask=torch.zeros(2,32,dtype=torch.bool);mask[0,0]=True;mask[1,1]=True
        zm,m,_=model(x,query='message',mask=mask,trace=True)
        zn,n,_=model(x,query='node',mask=mask,trace=True)
        assert b['s1'][0,SITE,0]==1 and b['s1'][1,SITE,1]==0
        assert b['s2'][:,-1].sum()>0
        assert trace_error(m,b,fields=('u1','s1','v1'))==0
        state_difference=float((n['v1'][0,SITE:]-m['v1'][0,SITE:]).abs().max())
        assert state_difference>0 and float(abs(zn[0]-zm[0]).max())>0
        assert trace_error(n,b,rows=torch.tensor([False,True]))==0
        assert torch.equal(m['m1'][:,SITE],n['m1'][:,SITE])
        readout_error=0.;active_readout_effect=0.
        rates=b['s2'].mean(1)
        for unit in range(32):
            changed=rates.clone();amount=b['s2'][:,-1,unit]
            changed[:,unit]-=amount/STEPS
            zz=model.readout(changed)
            observed=(z[:,0]-z[:,1])-(zz[:,0]-zz[:,1])
            expected=amount*(model.readout.weight[0,unit]-model.readout.weight[1,unit])/STEPS
            readout_error=max(readout_error,float(abs(observed-expected).max()))
            active_readout_effect=max(active_readout_effect,float(abs(expected).max()))
        assert readout_error<=TOL and active_readout_effect>0
    gradients=[];tangent_errors=[];forward_errors=[]
    for query in QUERIES:
        zz,tt,leaf=model(x,query=query,leaf=True,trace=True)
        gradient=torch.autograd.grad((zz[:,0]-zz[:,1]).sum(),leaf)[0]
        with torch.no_grad():
            target=tangent(model,b,torch.zeros(2,dtype=torch.long),torch.ones(2,dtype=torch.long),query)
        tangent_errors.append(float(abs(gradient-target).max()))
        forward_errors.append(max(float(abs(zz.detach()-z).max()),trace_error(tt,b)))
        gradients.append(gradient)
    gradient_difference=float(abs(gradients[0]-gradients[1]).max())
    np.savez_compressed(run.out/'dummy-witness.npz',input=x.numpy(),baseline=z.numpy(),
        message=zm.numpy(),node=zn.numpy(),message_gradient=gradients[0].numpy(),node_gradient=gradients[1].numpy(),
        **{'baseline_'+key:value.numpy() for key,value in b.items()})
    assert max(tangent_errors)<=TOL and max(forward_errors)==0 and gradient_difference>0
    return dict(active_selected=1,inactive_selected=1,active_final_readout_spikes=int(b['s2'][:,-1].sum()),
        first_layer_state_difference=state_difference,output_difference=float(abs(zn[0]-zm[0]).max()),
        active_final_readout_effect=active_readout_effect,final_readout_error=readout_error,
        tangent_errors=tangent_errors,leaf_forward_errors=forward_errors,local_graph_gradient_difference=gradient_difference,
        separate_witness_model_calls=5,separate_witness_forwarded_examples=10,separate_readout_calls=32,
        note='Hand-set dummy weights and zero inputs only. No study data or trained model used.')


def dummy(run):
    torch.manual_seed(99113)
    x=torch.rand(239,64,dtype=torch.float64);y=torch.arange(239)%10
    model=initialize(99117);run.seed=99117
    start=time.monotonic();run.train(model,x,y,99117,epochs=5);train_seconds=time.monotonic()-start
    start=time.monotonic()
    with torch.no_grad():
        for begin in range(0,133,BATCH):run.forward(model,x[begin:min(begin+BATCH,133)])
    test_seconds=time.monotonic()-start
    start=time.monotonic();panel(run,model,x[:120],y[:120],np.arange(120),99117);panel_seconds=time.monotonic()-start
    witness=nonvacuous_dummy(run)
    projected=2*(train_seconds*611680/(5*239)+test_seconds*7188/133+panel_seconds*4)
    assert projected<CAP
    return dict(scientific_model_outputs_collected=False,training_batches=[128,111],test_batches=[128,5],panel_batch=120,
        training_seconds=train_seconds,test_seconds=test_seconds,panel_seconds=panel_seconds,
        projected_seconds_with_2x_safety=projected,dummy_nonvacuity=witness,
        control_rows=run.tables['controls'].count,control_maxima=run.maxima,calls=run.calls,examples=run.examples)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--lock-commit')
    ap.add_argument('--dummy',action='store_true');args=ap.parse_args()
    for package,wanted in PACKAGES.items():assert version(package)==wanted
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    if not args.dummy:
        assert args.lock_commit,'A public lock is required.'
        lock_check(args.lock_commit)
    run=Run(args.output,args.dummy)
    metadata=dict(mode='dummy' if args.dummy else 'fixed-study',packages=PACKAGES,python=platform.python_version(),
        dtype='float64',threads=2,interop_threads=1,mkldnn=False,deterministic=True,lock_commit=args.lock_commit,
        plan_sha256=sha(ROOT/'analysis-plan.md'),runner_sha256=sha(__file__),inputs_sha256=sha(ROOT/'inputs.json'))
    dump(run.out/'metadata.json',metadata)
    try:
        if args.dummy:
            result=dummy(run);result.update(metadata);dump(run.out/'dummy-preflight.json',result)
            print(json.dumps(result,indent=2))
        else:
            (train,test),ids=data();tx,ty=train;xx,yy=test;summaries=[]
            for seed in SEEDS:
                run.seed=seed;model=initialize(seed);run.active_model=model;initial=parameter_hash(model)
                orders=run.train(model,tx,ty,seed)
                np.savez_compressed(run.out/f'schedule-{seed}.npz',orders=orders)
                checkpoint(run.out/f'model-{seed}.npz',model)
                with torch.no_grad():
                    full=torch.cat([run.forward(model,xx[start:start+BATCH])[0] for start in range(0,len(xx),BATCH)])
                np.savez_compressed(run.out/f'test-{seed}.npz',logits=full.numpy(),labels=yy.numpy())
                score=panel(run,model,xx[ids],yy[ids],ids,seed)
                accuracy=float((full.argmax(-1)==yy).double().mean())
                score.update(test_accuracy=accuracy,ability_gate=accuracy>=.8,
                    initial_parameter_sha256=initial,final_parameter_sha256=parameter_hash(model))
                summaries.append(score);dump(run.out/'model-summaries.json',summaries)
                print(f'Finished fixed seed {seed}; all cases retained.',flush=True)
            result=final_summary(summaries)
            result.update(calls=run.calls,examples=run.examples,training_updates=run.updates,
                control_rows=run.tables['controls'].count,control_maxima=run.maxima,
                elapsed_seconds=time.monotonic()-run.started)
            dump(run.out/'summary.json',result)
            assert (run.calls,run.examples,run.updates)==(5160,654868,4800)
            expected={'training':4800,'controls':1688,'cases':480,'events':30720,'regrets':1920,'joints':2880}
            assert {k:v.count for k,v in run.tables.items()}==expected
            run.budget();lock_check(args.lock_commit);run.budget()
        run.close()
        dump(run.out/'manifest.json',dict(status='complete',files={p.relative_to(run.out).as_posix():sha(p)
             for p in sorted(run.out.rglob('*')) if p.is_file()}))
    except BaseException as error:
        if run.active_model is not None:checkpoint(run.out/'partial-model.npz',run.active_model)
        if run.latest is not None:
            np.savez_compressed(run.out/'failed-forward.npz',**{k:v.numpy() for k,v in run.latest.items()})
        dump(run.out/'failure.json',dict(type=type(error).__name__,message=str(error),seed=run.seed,
            calls=run.calls,examples=run.examples,updates=run.updates,elapsed_seconds=time.monotonic()-run.started))
        run.close()
        dump(run.out/'manifest.json',dict(status='failed',files={p.relative_to(run.out).as_posix():sha(p)
             for p in sorted(run.out.rglob('*')) if p.is_file() and p.name!='manifest.json'}))
        raise


if __name__=='__main__':main()

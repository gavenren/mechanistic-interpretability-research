"""Audit saved Experiment 2A arrays. No trained-model forward or search."""
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
RESULTS = STUDY / 'results'
START = None
NP = None


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def peak():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def guard():
    if time.monotonic()-START >= 59:
        raise TimeoutError('Saved-result audit time limit')
    need(peak() <= 4*1024**3, 'Saved-result audit memory limit')
    try:
        permit = read(ROOT/'.local-state/edit-procedure-permit.json')
        expiry = permit['expires_at']; now = time.time()
        ok = permit['allowed'] is True and type(expiry) in (int,float) and math.isfinite(expiry) and now < expiry <= now+120
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        ok = False
    need(ok, 'Usage permit is missing, invalid, or expired')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            guard(); result.update(block)
    return result.hexdigest()


def close(a, b, tol=1e-10):
    a, b = NP.asarray(a), NP.asarray(b)
    need(a.shape == b.shape and NP.isfinite(a).all() and NP.isfinite(b).all(), 'Invalid compared array')
    error = float(NP.max(NP.abs(a-b))) if a.size else 0.
    need(error <= tol, 'Saved value differs from recalculation')
    return error


def arrays(path):
    guard()
    with NP.load(path, allow_pickle=False) as archive:
        return {name:archive[name].copy() for name in archive.files}


def check_manifest(folder):
    saved = read(folder/'manifest.json')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()
              and p.name != 'manifest.json' and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
    need(actual == set(saved), 'Manifest file list differs')
    for relative, expected in saved.items():
        path = (folder/relative).resolve()
        need(path.is_relative_to(folder.resolve()), 'Manifest path leaves output folder')
        need(digest(path) == expected, 'Manifest hash differs: '+relative)
    return {'files':len(saved), 'sha256':digest(folder/'manifest.json')}


def summary_measures(kind, rows, p, edited, intact):
    base, h, labels, mask = intact['probabilities'], intact['hidden'], intact['labels'], intact['mask']
    target = rows[:,2] if kind == 'donors' else rows[:,1] if kind == 'controls' else rows[:,3]
    values = {'tv_target':NP.abs(p-base[target]).sum(1)/2,
              'correct_target':(p.argmax(1)==labels[target]).astype(float),
              'changed_answer':(p.argmax(1)!=base[rows[:,0]].argmax(1)).astype(float)}
    if kind == 'donors':
        close(edited,h[rows[:,1]][:,mask]); columns=(3,4,5,6,7,8)
        identity=rows[:,0]==rows[:,1]
        close(p[identity],base[rows[identity,0]])
    elif kind == 'controls':
        columns=(2,3,4,5); expected=h[rows[:,0]][:,mask].copy()
        zero=rows[:,5]==1; coherent=rows[:,5]==2; noop=rows[:,5]==0
        expected[zero]=0; expected[coherent]=h[rows[coherent,1]][:,mask]
        close(edited,expected); close(p[noop],base[rows[noop,0]]); close(p[coherent],base[target[coherent]])
    else:
        columns=(5,6,7,8,9,10)
        need(NP.array_equal(rows[::2,:10],rows[1::2,:10]), 'Order pair ledger differs')
        one, two=h[rows[:,1]][:,mask],h[rows[:,2]][:,mask]
        last=NP.where(rows[:,10]==0,rows[:,2],rows[:,1]); close(edited,h[last][:,mask])
        values.update(tv_second_target=NP.abs(p-base[rows[:,4]]).sum(1)/2,
                      correct_second_target=(p.argmax(1)==labels[rows[:,4]]).astype(float),
                      first_request_met=(NP.max(NP.abs(edited-one),axis=1)<=1e-10).astype(float),
                      second_request_met=(NP.max(NP.abs(edited-two),axis=1)<=1e-10).astype(float),
                      coordinate_requests_compatible=(NP.max(NP.abs(one-two),axis=1)<=1e-10).astype(float),
                      order_probability_distance=NP.repeat(NP.abs(p[::2]-p[1::2]).sum(1)/2,2),
                      order_answer_disagreement=NP.repeat(p[::2].argmax(1)!=p[1::2].argmax(1),2).astype(float))
        need(NP.all(NP.where(rows[:,10]==0,values['second_request_met'],values['first_request_met'])==1), 'Last-write control failed')
    return values, columns


def selector():
    folder=RESULTS/'selector'; ledger=arrays(folder/'pair-ledgers.npz'); summary=read(folder/'summary.json')
    counts={'donors':36864,'controls':2304,'orders':12288}; total=0; max_error=0.; batches=0
    donor_table=[]; order_table=[]; control_table=[]
    for seed in range(8):
        intact=arrays(folder/f'intact-seed-{seed}.npz')
        need(intact['states'].shape==(192,4) and intact['probabilities'].shape==(192,4),'Selector intact shapes differ')
        need(NP.array_equal(intact['labels'],NP.where(intact['states'][:,2]==0,intact['states'][:,0],intact['states'][:,1])), 'Selector labels differ')
        close(intact['probabilities'],intact['reference_probabilities'])
        accuracy=float((intact['probabilities'].argmax(1)==intact['labels']).mean()); need(accuracy>=.99,'Selector intact gate failed')
        stored=read(folder/f'summary-seed-{seed}.json')
        close(accuracy,stored['controls']['intact_accuracy']); need(stored['controls']['passed'] is True,'Selector model control failed')
        expected_groups={(r['kind'],tuple(r['group'])):r for r in stored['groups']}
        for kind, count in counts.items():
            data=[]
            for offset in range(0,count,768):
                a=arrays(folder/f'{kind}-seed-{seed}-row-{offset:05d}.npz')
                need(NP.array_equal(a['rows'],ledger[kind][offset:offset+768]),'Selector saved ledger differs')
                p=a['probabilities']; need(p.shape==(len(a['rows']),4),'Selector probability shape differs')
                need(NP.isfinite(p).all() and NP.all(p>=0),'Invalid selector probabilities'); close(p.sum(1),NP.ones(len(p)))
                r=read(folder/f'{kind}-seed-{seed}-row-{offset:05d}-audit.json')
                need(r['passed'] is True and r['rows']==len(p),'Selector batch replay did not pass')
                need(r['absolute_tolerance']==1e-10,'Selector tolerance differs')
                e=max(r['maximum_mask_replay_error'],r['maximum_probability_replay_error']); need(e<=1e-10,'Selector recorded replay failed')
                max_error=max(max_error,e); batches+=1; data.append(a)
            rows=NP.concatenate([a['rows'] for a in data]); p=NP.concatenate([a['probabilities'] for a in data]); edited=NP.concatenate([a['edited_mask'] for a in data])
            need(len(rows)==count,'Selector row count differs'); total+=len(rows)
            values, columns=summary_measures(kind,rows,p,edited,intact)
            for group in NP.unique(rows[:,columns],axis=0):
                take=NP.all(rows[:,columns]==group,axis=1); original=expected_groups.pop((kind,tuple(map(int,group))))
                need(original['n']==int(take.sum()),'Selector group count differs')
                for name,value in values.items(): close(float(value[take].mean()),original[name])
            if kind=='donors':
                need(int(NP.sum(rows[:,3]==0))==18432 and int(NP.sum(rows[:,3]==1))==18432,'Selector split differs')
                for same,active in itertools.product((0,1),repeat=2):
                    take=(rows[:,3]==1)&(rows[:,4]==same)&(rows[:,5]==active)
                    donor_table.append(dict(seed=seed,same_a=bool(same),active=bool(active),n=int(take.sum()),
                                           **{name:float(value[take].mean()) for name,value in values.items()}))
            elif kind=='controls':
                for same,active,operation in itertools.product((0,1),(0,1),(0,1,2)):
                    take=(rows[:,2]==1)&(rows[:,3]==same)&(rows[:,4]==active)&(rows[:,5]==operation)
                    control_table.append(dict(seed=seed,same_a=bool(same),active=bool(active),operation=operation,n=int(take.sum()),
                                             **{name:float(value[take].mean()) for name,value in values.items()}))
            else:
                for equal,active in itertools.product((0,1),repeat=2):
                    take=(rows[:,5]==1)&(rows[:,6]==equal)&(rows[:,7]==active)
                    order_table.append(dict(seed=seed,requested_values_equal=bool(equal),active=bool(active),n=int(take.sum()),
                                           **{name:float(value[take].mean()) for name,value in values.items()}))
        need(not expected_groups,'Unchecked selector summary group')
    need(total==411648==summary['rows'] and summary['models']==8 and summary['passed'] is True,'Selector summary count differs')
    return dict(rows=total,models=8,batches_checked=batches,maximum_recorded_replay_error=max_error,
                test_donor_by_model=donor_table,test_order_by_model=order_table,test_controls_by_model=control_table)


def codes():
    folder=RESULTS/'codes'; summary=read(folder/'codes-summary.json'); controls=read(folder/'codes-controls.json')
    with (folder/'codes-cases.csv').open() as stream: stored=list(csv.DictReader(stream))
    need(len(stored)==288 and len(controls)==6,'Code case count differs')
    lookup={(int(r['seed']),r['split'],r['background_id'],int(r['bit'])):r for r in stored}; rows=[]; maximum_error=0.
    for seed in range(6):
        a=arrays(folder/f'codes-seed-{seed}.npz'); n=len(a['h']); need(n==48,'Code model case count differs')
        need(a['baseline_code'].shape==(48,24) and a['h'].shape==(48,12),'Code shapes differ')
        need(list(a['split'])==['development']*16+['test']*32 and list(a['bit'])==[0,1]*24,'Code split or bit order differs')
        close(a['squared_error_budget'],NP.sum((a['baseline_reconstruction']-a['h'])**2,axis=1),1e-9)
        for name in ('baseline','low','high','witness','same'):
            code=a[name+'_code']; reconstruction=a[name+'_reconstruction']; probability=a[name+'_probability']
            need(NP.isfinite(code).all() and NP.all(code>=0) and NP.all((code>0).sum(1)<=4),'Code constraint failed')
            error=NP.sum((reconstruction-a['h'])**2,axis=1)
            need(NP.all(error<=a['squared_error_budget']+1e-9*NP.maximum(1.,a['squared_error_budget'])),'Code error limit failed')
            logits=8*(reconstruction@a['rare_direction']/.25-.5)
            independent=NP.exp(-NP.logaddexp(0.,-logits)); close(probability,independent,1e-9)
            if name!='baseline': close(error,a[name+'_squared_error'],1e-9)
        for name,mask_name in (('witness','alternative_found'),('same','same_vector_found')):
            mask=a[mask_name]; distance=NP.max(NP.abs(a[name+'_code']-a['baseline_code']),axis=1)
            need(NP.all(distance[mask]>1e-7) and NP.all(distance[~mask]==0),'Code witness flags differ')
        mask=a['same_vector_found']; need(NP.all(NP.max(NP.abs(a['same_reconstruction']-a['baseline_reconstruction']),axis=1)[mask]<=1e-9),'Same-vector claim failed')
        span=a['high_probability']-a['low_probability']; need(NP.all(span>=-1e-12),'Negative code span')
        need(NP.all(a['low_probability']<=a['baseline_probability']+1e-12) and NP.all(a['high_probability']>=a['baseline_probability']-1e-12),'Code span omits baseline')
        control=controls[seed]; need(control['seed']==seed and control['status']=='pass','Code recorded control failed')
        for name in ('encoder_replay_error','no_change_error','decoded_replay_error','saved_file_replay_error'):
            need(control[name]<=1e-9,'Code recorded replay failed'); maximum_error=max(maximum_error,control[name])
        for i in range(n):
            r=lookup.pop((seed,str(a['split'][i]),str(a['sample_id'][i]),int(a['bit'][i])))
            for name,value in [('baseline_squared_error',a['squared_error_budget'][i]),('low_squared_error',a['low_squared_error'][i]),
                               ('high_squared_error',a['high_squared_error'][i]),('baseline_probability',a['baseline_probability'][i]),
                               ('low_probability',a['low_probability'][i]),('high_probability',a['high_probability'][i]),('probability_span',span[i])]: close(float(r[name]),value,1e-9)
            need(int(r['valid_candidates'])==int(a['candidate_counts'][i].sum()),'Code candidate count differs')
            need((r['alternative_found']=='True')==bool(a['alternative_found'][i]) and (r['same_vector_found']=='True')==bool(mask[i]),'CSV witness flag differs')
            rows.append(dict(seed=seed,split=str(a['split'][i]),alternative=bool(a['alternative_found'][i]),same_vector=bool(mask[i]),span=float(span[i])))
    need(not lookup,'Unchecked code CSV case'); by_split={}; by_model=[]
    for split in ('development','test'):
        selected=[r for r in rows if r['split']==split]; spans=[r['span'] for r in selected]
        actual=dict(cases=len(selected),alternatives_found=sum(r['alternative'] for r in selected),same_vector_found=sum(r['same_vector'] for r in selected),
                    mean_probability_span=float(NP.mean(spans)),maximum_probability_span=max(spans))
        for name,value in actual.items(): close(value,summary['by_split'][split][name])
        by_split[split]=actual
        for seed in range(6):
            v=[r['span'] for r in selected if r['seed']==seed]
            by_model.append(dict(seed=seed,split=split,mean_probability_span=float(NP.mean(v)),maximum_probability_span=max(v)))
    return dict(cases=288,by_split=by_split,by_model=by_model,maximum_recorded_replay_error=maximum_error)


def normalization():
    folder=RESULTS/'normalization'; inputs=read(folder/'inputs.json'); summary=read(folder/'status.json')
    parameters=arrays(folder/'layernorm_parameters.npz'); gaps=read(folder/'rule_gaps.json'); lookup={(r['prompt_id'],r['site'],r['alpha']):r for r in gaps}
    need(len(gaps)==1024 and summary['completed_prompt_ids']==list(range(32)),'Normalization case count differs')
    effects=[]; controls_max=0.; recorded_full_max=0.; checked=0
    for pid in range(32):
        a=arrays(folder/f'prompt_{pid:02d}.npz'); checks=read(folder/f'prompt_{pid:02d}_checks.json')
        need(len(checks)==64,'Normalization check count differs'); check_map={(r['site'],r['alpha'],r['mode']):r for r in checks}
        for site in range(8):
            x=a[f'x_{site}']; base_y=a[f'y_{site}']; gamma=parameters[f'gamma_{site}']; beta=parameters[f'beta_{site}']; eps=float(parameters[f'eps_{site}'])
            need(x.shape==base_y.shape==gamma.shape==beta.shape==(128,) and eps==1e-5,'Normalization parameters differ')
            mean=float(x.mean()); variance=float(NP.mean((x-mean)**2))
            for alpha in (0.,.1,.5,1.):
                edited=x.copy(); edited[0]+=alpha*math.sqrt(variance)*math.sqrt(128); values={}
                for mode in ('recomputed','frozen'):
                    key=f's{site}_a{alpha:g}_{mode}'; y=a[key+'_y']; logits=a[key+'_logits']; r=check_map.pop((site,alpha,mode))
                    close(a[key+'_input'],edited,1e-8)
                    m,v=(float(edited.mean()),float(NP.mean((edited-edited.mean())**2))) if mode=='recomputed' else (mean,variance)
                    close(y,(edited-m)/math.sqrt(v+eps)*gamma+beta,1e-8)
                    close(y,a[key+'_replay_y'],1e-8); close(logits,a[key+'_replay_logits'],1e-8)
                    for name in ('layernorm_replay_max_abs','full_logit_replay_max_abs','selected_effect_replay_max_abs','no_change_max_abs'):
                        need(0<=r[name]<=1e-8,'Normalization recorded check failed')
                    recorded_full_max=max(recorded_full_max,r['full_logit_replay_max_abs']); checked+=1
                    close(float(NP.linalg.norm(y-base_y)),r['layernorm_effect_l2'])
                    close(float(NP.max(NP.abs(logits-a['baseline_logits']))),r['selected_score_effect_max_abs'])
                    if alpha==0:
                        controls_max=max(controls_max,close(y,base_y,1e-8),close(logits,a['baseline_logits'],1e-8))
                    values[mode]=(y,logits)
                dy=values['recomputed'][0]-values['frozen'][0]; ds=values['recomputed'][1]-values['frozen'][1]
                row=dict(prompt_id=pid,site=site,alpha=alpha,layernorm_rule_gap_l2=float(NP.linalg.norm(dy)),
                         layernorm_rule_gap_max_abs=float(NP.max(NP.abs(dy))),selected_score_rule_gap_max_abs=float(NP.max(NP.abs(ds))))
                original=lookup.pop((pid,site,alpha))
                for name in ('layernorm_rule_gap_l2','layernorm_rule_gap_max_abs','selected_score_rule_gap_max_abs'): close(row[name],original[name])
                effects.append(row)
        need(not check_map,'Unchecked normalization control')
    need(not lookup,'Unchecked normalization effect'); token_sum=sum(len(p['token_ids']) for p in inputs['prompts'])
    need(summary['forwards']==4128 and summary['tokens']==129*token_sum,'Normalization forward or token count differs')
    by_size={}
    for alpha in (.1,.5,1.):
        rows=[r for r in effects if r['alpha']==alpha]
        by_size[str(alpha)]=dict(cases=len(rows),mean_normalization_gap_l2=float(NP.mean([r['layernorm_rule_gap_l2'] for r in rows])),
                               maximum_normalization_gap_l2=max(r['layernorm_rule_gap_l2'] for r in rows),
                               median_selected_score_gap=float(NP.median([r['selected_score_rule_gap_max_abs'] for r in rows])),
                               maximum_selected_score_gap=max(r['selected_score_rule_gap_max_abs'] for r in rows))
    return dict(scientific_forwards=4128,scientific_tokens=summary['tokens'],recorded_checks=checked,
                zero_edit_maximum_saved_coordinate_error=controls_max,maximum_recorded_full_logit_replay_error=recorded_full_max,by_edit_size=by_size,
                limitation='The full vocabulary was not saved. Its replay maxima are checked as recorded. No transformer forward was repeated.')


def compact_groups(rows, keys, metrics):
    result=[]
    for values in sorted({tuple(r[k] for k in keys) for r in rows}):
        chosen=[r for r in rows if tuple(r[k] for k in keys)==values]
        item=dict(zip(keys,values)); item['models']=len(chosen)
        for metric in metrics:
            vals=[r[metric] for r in chosen]
            item[metric+'_mean_across_models']=float(NP.mean(vals)); item[metric+'_min_model']=min(vals); item[metric+'_max_model']=max(vals)
        result.append(item)
    return result


def main():
    global START,NP
    START=time.monotonic(); out=STUDY/'analysis/audit.json'
    need(not out.exists(),'Audit receipt already exists')
    result={'status':'stopped','new_model_forwards':0,'new_training':False,'search_repeated':False}
    with (ROOT/'.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard()
        try:
            import numpy as np
            NP=np
            science=read(RESULTS/'status.json'); prep=read(STUDY/'preparation/status.json')
            need(science['status']=='complete' and prep['status']=='passed','Source execution incomplete')
            need(science['runtime']==prep['runtime'],'Runtime changed after preparation')
            need(science['runtime']['numpy']=='2.3.5' and science['runtime']['torch']=='2.8.0' and science['runtime']['transformers']=='4.57.6','Runtime version differs')
            result['manifests']={name:check_manifest(STUDY/name) for name in ('preparation','results')}
            sources=read(STUDY/'input-manifest.json')
            for relative,expected in sources['files'].items(): need(digest(ROOT/relative)==expected,'Source hash differs: '+relative)
            result['source_files_checked']=len(sources['files']); result['source_commit']=sources['source_commit']
            result['plan_lock']=prep['plan_lock']; result['execution_lock']=science['plan_lock']
            result['selector']=selector(); result['codes']=codes(); result['normalization']=normalization()
            result['selector']['test_donor_summary']=compact_groups(result['selector']['test_donor_by_model'],('same_a','active'),('correct_target','tv_target','changed_answer'))
            result['selector']['test_order_summary']=compact_groups(result['selector']['test_order_by_model'],('requested_values_equal','active'),('order_probability_distance','order_answer_disagreement','coordinate_requests_compatible'))
            need(science['forwarded_tokens']==prep['forwarded_tokens']+result['normalization']['scientific_tokens']==33536,'Cumulative token count differs')
            need(prep['forwarded_tokens']==512,'Preparation token count differs')
            result.update(status='passed',forwarded_tokens_in_source_run=science['forwarded_tokens'],audit_code_sha256=digest(Path(__file__)),
                          limits='Saved-array check only. Trained encoder, decoder, and transformer forwards were not repeated. Numeric pass is not semantic proof.')
            guard()
        except Exception as exc:
            result.update(status='stopped',error_type=type(exc).__name__,reason=str(exc).replace(str(ROOT),'[repository]'))
        finally:
            result.update(measured_seconds_before_receipt=time.monotonic()-START,peak_memory_bytes=peak(),
                          closing_allowance_seconds=1.,conservative_stage_charge_seconds=60.)
            if result['measured_seconds_before_receipt']+1>60: result.update(status='stopped',reason='Audit cap reached during saving')
            closing=time.monotonic(); out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
            if time.monotonic()-closing>1:
                result.update(status='stopped',reason='Audit receipt exceeded closing allowance',measured_seconds_before_receipt=time.monotonic()-START)
                out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('status','measured_seconds_before_receipt','peak_memory_bytes')}) )
        return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

"""Dummy-only full protocol and non-vacuous stop-rule checks."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time
import numpy as np
import intervals as iv
from protocol import refine,primary_counts,sign_reversal,classify
from run import ROOT,Record,setup,train,evaluate,dump,sha,manifest
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    setup();record=Record(args.output);out=args.output;began=time.monotonic()
    receipt=dict(status='running',scientific_model_outputs_collected=False,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'code').glob('*.py'))},
        plan_sha256=sha(ROOT/'analysis-plan.md'),numpy=np.__version__,torch=torch.__version__)
    cases=[]
    try:
        # This table checks strict signs, the small rule, and cap precedence.
        lo=np.array([1e-9,-2e-9,-1e-8,-1.]);hi=np.array([2e-9,-1e-9,1e-8,1.])
        status=classify(lo,hi,0,0)
        record.control('stop_rule_priority',status.tolist()==['positive','negative','small','unresolved'],status.tolist())
        record.control('zero_primary_denominator',primary_counts(np.zeros(2),np.full(2,-1.),np.ones(2))['primary_fraction'] is None)
        p=primary_counts(np.array([1.,-1.,0.,2.]),np.array([.1,-2.,-1.,-1.]),np.array([2.,1.,1.,1.]))
        record.control('nonzero_primary_denominator',p['nonzero_ordinary']==3 and p['unresolved_nonzero']==2 and p['primary_fraction']==2/3,p)
        record.control('reversal_only_certified',sign_reversal(np.array([-1.,1.,0.,1.,-1.]),
            np.array(['positive','negative','positive','small','unresolved'])).tolist()==[True,True,False,False,False])
        # Each small case has a known exact equilibrium effect. All fractions are exact.
        specifications=[
            ('positive',np.zeros((2,2)),[.5,.25],[1.,0.],[0.,0.],[0.,0.],128,Fraction(1,2),'positive'),
            ('negative',np.zeros((2,2)),[.5,.25],[-1.,0.],[0.,0.],[0.,0.],128,Fraction(-1,2),'negative'),
            ('small',np.zeros((2,2)),[.5,.25],[0.,0.],[0.,0.],[0.,0.],128,Fraction(0),'small'),
            ('cap',np.diag([.875,.5]),[.125,.25],[1.,1.],[0.,0.],[0.,0.],0,Fraction(1),'unresolved'),
            ('full_cap',np.diag([.999,.5]),[.001,.25],[1.,0.],[0.,0.],[0.,0.],128,
             Fraction.from_float(.001)/(1-Fraction.from_float(.999)),'unresolved'),
            ('reversal',np.diag([.875,.5]),[.125,.25],[1.,1.],[0.,0.],[0.,100.],128,Fraction(1),'positive'),
        ]
        for name,w,aa,cc,bb,ee,cap,truth,wanted in specifications:
            path=out/name;path.mkdir();a=np.array([aa]);c=np.array(cc);base=np.array([bb]);edit=np.array([ee])
            readout=np.stack([c,np.zeros(2)]);q=iv.contraction_upper(w)
            initial,cost,final_status,steps,final=refine(base,edit,w,a,iv.point(a),np.array([0]),
                readout,np.array([0]),np.array([1]),q,path,record.control,record.tick,cap=cap)
            naive=np.sum(c*(base-edit),axis=1)
            record.control('known_stop_'+name,final_status[0]==wanted,dict(status=final_status[0],steps=int(steps[0])))
            for prefix,bounds in [('initial',initial),('final',final)]:
                contains=Fraction.from_float(float(bounds['lower'][0]))<=truth<=Fraction.from_float(float(bounds['upper'][0]))
                record.control('exact_effect_containment_'+name+'_'+prefix,contains)
            reversed_sign=bool(sign_reversal(naive,final_status)[0])
            record.control('known_reversal_'+name,reversed_sign==(name=='reversal'))
            cases.append(dict(name=name,naive=float(naive[0]),exact_effect=str(truth),status=final_status[0],
                extra_updates=int(steps[0]),certified_reversal=reversed_sign,**cost))
        # Full 32-coordinate model uses generated data only. No UCI file is opened.
        rng=np.random.default_rng(2026101603);tx=torch.from_numpy(rng.random((239,64)))
        ty=torch.arange(239,dtype=torch.int64)%10;path=out/'real-shapes';path.mkdir()
        weights,cost=train(99073,tx,ty,path,record,epochs=2)
        test=rng.random((133,64));labels=np.arange(133)%10;panel=np.arange(120)
        result=evaluate(99073,weights,test,labels,panel,path,record)
        record.control('real_shape_counts',cost['training_updates']==4 and result['effects']==3840 and
            np.load(path/'test.npz')['state'].shape==(133,32),dict(training_shapes=[128,111],
            test_shapes=[128,5],panel=120,pairs=3840,training_updates=cost['training_updates']))
        # Failure paths must preserve the failed check and partial state.
        failure=out/'intentional-failure';failure.mkdir()
        class ExpectedFailure(Exception):pass
        def reject(name,passed,evidence=None):
            dump(failure/'expected-control.json',dict(name=name,passed=False,evidence=evidence))
            raise ExpectedFailure('Intentional engineering check, not a study failure.')
        try:
            refine(np.zeros((1,2)),np.zeros((1,2)),np.zeros((2,2)),np.ones((1,2)),iv.point(np.ones((1,2))),
                np.array([0]),np.eye(2),np.array([0]),np.array([1]),0.,failure,reject,record.tick)
        except ExpectedFailure:pass
        record.control('partial_state_on_failure',(failure/'refinement-partial.npz').exists())
        try:Record(out)
        except AssertionError:refused=True
        else:refused=False
        record.control('existing_output_refused',refused)
        limited=Record(out/'time-limit',cap=-1)
        try:limited.tick()
        except TimeoutError:stopped=True
        else:stopped=False
        finally:limited.close()
        record.control('time_limit_enforced',stopped)
        receipt.update(status='passed',cases=cases,real_shapes=result,
            full_dummy_training_updates=cost['training_updates'])
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        record.close();receipt.update(elapsed_seconds=time.monotonic()-began,control_rows=record.controls)
        dump(out/'receipt.json',receipt);manifest(out)
    print(json.dumps(dict(status=receipt['status'],checks=record.controls,elapsed_seconds=receipt['elapsed_seconds'],
        cases=cases),indent=2))


if __name__=='__main__':main()

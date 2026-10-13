#!/usr/bin/env python3
"""Exact-batch post-failure diagnostic. Never trains or imports the study runner."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
LOCK='c406a08091cbc74caef1a1c4690aa29482667e0a'
TEST_HASH='6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'
EXPECTED_ERROR=2.1457672119140625e-5
TOL=1e-10


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def circular_pad(x):
    row=torch.arange(-1,x.shape[-2]+1)%x.shape[-2]
    col=torch.arange(-1,x.shape[-1]+1)%x.shape[-1]
    return x.index_select(-2,row).index_select(-1,col)


def circular_shift(x,amount):
    index=(torch.arange(x.shape[-1])-amount)%x.shape[-1]
    return x.index_select(-1,index)


class HookNet(nn.Module):
    def __init__(self,weights,dtype):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,padding=1,padding_mode='circular',dtype=dtype)
        self.conv2=nn.Conv2d(8,16,3,padding=1,padding_mode='circular',dtype=dtype)
        self.conv3=nn.Conv2d(16,32,3,padding=1,padding_mode='circular',dtype=dtype)
        self.fc=nn.Linear(32,10,dtype=dtype)
        self.relu1=nn.ReLU()
        self.register_buffer('kernel',torch.from_numpy(weights['kernel']).to(dtype))
        self.load_state_dict({k:torch.from_numpy(v).to(dtype) for k,v in weights.items()})
        self.relu1.register_forward_hook(self.remove_channel)

    def remove_channel(self,module,inputs,output):
        changed=output.clone(); changed[:,1]=0
        self.local={'removed':float(changed[:,1].abs().max()),
                    'untouched':float((changed[:,[0,2,3,4,5,6,7]]-output[:,[0,2,3,4,5,6,7]]).abs().max())}
        return changed

    def forward(self,x):
        trace={'input':x}
        a=self.relu1(self.conv1(x)); trace['edited_relu1']=a
        a=a[:,:,::2,::2]; trace['down1']=a
        a=F.relu(self.conv2(a)); trace['relu2']=a
        a=a[:,:,::2,::2]; trace['down2']=a
        a=F.relu(self.conv3(a)); trace['relu3']=a
        a=a.mean(dim=(-2,-1)); trace['pooled']=a
        a=self.fc(a); trace['logits']=a
        return trace,self.local


def explicit(weights,x):
    state={k:torch.from_numpy(v).to(x.dtype) for k,v in weights.items()}
    def conv(a,name):
        height,width=a.shape[-2:]
        windows=F.unfold(circular_pad(a),kernel_size=3)
        out=state[name+'.weight'].reshape(-1,windows.shape[1]) @ windows
        out=out+state[name+'.bias'][None,:,None]
        return out.reshape(len(a),-1,height,width)
    trace={'input':x}
    a=F.relu(conv(x,'conv1'))
    mask=torch.ones(1,8,1,1,dtype=x.dtype);mask[:,1]=0
    changed=a*mask
    local={'removed':float(changed[:,1].abs().max()),
           'untouched':float((changed[:,[0,2,3,4,5,6,7]]-a[:,[0,2,3,4,5,6,7]]).abs().max())}
    a=changed;trace['edited_relu1']=a
    a=a[:,:,::2,::2];trace['down1']=a
    a=F.relu(conv(a,'conv2'));trace['relu2']=a
    a=a[:,:,::2,::2];trace['down2']=a
    a=F.relu(conv(a,'conv3'));trace['relu3']=a
    a=a.sum(-1).sum(-1)/(a.shape[-1]*a.shape[-2]);trace['pooled']=a
    a=a @ state['fc.weight'].T+state['fc.bias'];trace['logits']=a
    return trace,local


def errors(trace,other):
    amounts={'input':-4,'edited_relu1':-4,'down1':-2,'relu2':-2,'down2':-1,'relu3':-1}
    result={}
    for name,left in trace.items():
        reference=circular_shift(left,amounts[name]) if name in amounts else left
        delta=(other[name]-reference).abs().numpy()
        result[name]={'maximum_error':float(delta.max()),
                      'argmax':list(map(int,np.unravel_index(delta.argmax(),delta.shape)))}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--expected-plan-sha256',required=True)
    ap.add_argument('--expected-code-sha256',required=True)
    args=ap.parse_args()
    plan=ROOT/'reviews/failure-diagnostic-plan.md'
    assert sha(plan)==args.expected_plan_sha256 and sha(__file__)==args.expected_code_sha256
    assert version('numpy')=='2.3.5' and version('torch')=='2.8.0'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    receipt={'status':'running','label':'post-failure diagnostic; original experiment remains failed',
             'plan_sha256':sha(plan),'code_sha256':sha(__file__),'original_lock':LOCK,
             'python':platform.python_version(),'numpy':version('numpy'),'torch':version('torch')}
    checks=[];outputs={};diagnostics={}
    def check(name,error,tolerance):
        checks.append({'name':name,'error':float(error),'tolerance':tolerance,
                       'passed':bool(np.isfinite(error) and error<=tolerance)})
    try:
        manifest=json.loads((args.results/'manifest.json').read_text())
        assert manifest['status']=='failed'
        for name,digest in manifest['files'].items(): assert sha(args.results/name)==digest
        repo=ROOT.parents[1]
        for name in ['analysis-plan.md','code/run.py','requirements.txt','data/optdigits.tra','data/optdigits.tes']:
            path=ROOT/name;rel=path.relative_to(repo).as_posix()
            locked=subprocess.run(['git','show',LOCK+':'+rel],cwd=repo,capture_output=True,check=True).stdout
            assert locked==path.read_bytes()
        failure=json.loads((args.results/'failure.json').read_text())
        assert failure['context']=={'architecture':'direct','batch':6,'channel':1,'seed':2,'shift':'0,-4'}
        assert failure['control_maxima']['four_pixel_logits']==EXPECTED_ERROR
        raw=ROOT/'data/optdigits.tes';assert sha(raw)==TEST_HASH
        data=np.loadtxt(raw,delimiter=',',dtype=np.int64)[768:896,:64].reshape(128,1,8,8)
        original=np.pad(data.astype(np.float32)/16,((0,0),(0,0),(4,4),(4,4)))
        shifted=original[:,:,:,(np.arange(16)+4)%16].copy()
        weights=dict(np.load(args.results/'models/2-direct.npz'))
        saved=np.load(args.results/'logits/2-direct.partial.npy',mmap_mode='r')[0,2,768:896].copy()
        for dtype_name,dtype in [('float32',torch.float32),('float64',torch.float64)]:
            for path in ('hook','explicit'):
                if time.monotonic()-start>300: raise TimeoutError('Five-minute diagnostic cap reached')
                traces=[]
                model=HookNet(weights,dtype).eval() if path=='hook' else None
                with torch.no_grad():
                    for label,array in [('original',original),('shifted',shifted)]:
                        tensor=torch.from_numpy(array).to(dtype)
                        trace,local=model(tensor) if model is not None else explicit(weights,tensor)
                        assert all(torch.isfinite(value).all() for value in trace.values())
                        traces.append(trace)
                        outputs[path+'_'+dtype_name+'_'+label]=trace['logits'].numpy()
                        for kind,error in local.items():check(path+'_'+dtype_name+'_'+label+'_'+kind,error,0.)
                diagnostics[path+'_'+dtype_name]=errors(*traces)
        check('hook_float32_saved_replay',np.max(np.abs(outputs['hook_float32_original']-saved)),TOL)
        reproduced=diagnostics['hook_float32']['logits']['maximum_error']
        check('recorded_failure_maximum_reproduced',abs(reproduced-EXPECTED_ERROR),TOL)
        for path in ('hook','explicit'):
            check(path+'_float64_logit_symmetry',diagnostics[path+'_float64']['logits']['maximum_error'],TOL)
        for label in ('original','shifted'):
            check('float64_implementation_agreement_'+label,
                  np.max(np.abs(outputs['hook_float64_'+label]-outputs['explicit_float64_'+label])),TOL)
        np.savez_compressed(args.output/'batch-logits.npz',**outputs,saved_float32_original=saved)
        receipt.update({'status':'passed' if all(c['passed'] for c in checks) else 'failed',
                        'original_failure_remains':True,'batch_rows':[768,895], 'channel':1,
                        'shift':[0,-4],'model_sha256':sha(args.results/'models/2-direct.npz'),
                        'raw_data_sha256':TEST_HASH,'checks':checks,'stage_symmetry':diagnostics,
                        'model_training_performed':False,'evaluated_batch_conditions':8})
        if time.monotonic()-start>300: raise TimeoutError('Five-minute diagnostic cap reached')
    except BaseException as error:
        receipt.update({'status':'execution_failed','error':str(error),'error_type':type(error).__name__,
                        'checks':checks,'stage_symmetry':diagnostics})
        if outputs:np.savez_compressed(args.output/'partial-batch-logits.npz',**outputs)
        raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-start
        dump(args.output/'receipt.json',receipt)
        dump(args.output/'manifest.json',{p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

"""Dummy-only full-shape timing. No study input files are read."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
import intervals as iv
from model import Model,optimizer,ordinary_solve,mapping


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
    started=time.monotonic();rng=np.random.default_rng(2026101602)
    model=Model(99071);opt=optimizer(model)
    x=torch.from_numpy(rng.uniform(size=(239,64)));y=torch.arange(239)%10
    start=time.monotonic()
    for epoch in range(10):
        for begin in (0,128):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x[begin:begin+128]),y[begin:begin+128])
            assert torch.isfinite(loss);loss.backward();opt.step();model.project()
    training_seconds=time.monotonic()-start
    a=model.arrays();w=a['recurrent.weight'];v=a['readout.weight'];q=iv.contraction_upper(w)
    dummy=rng.uniform(size=(133,64));inj=dummy@a['input.weight'].T+a['input.bias']
    start=time.monotonic();test=[]
    for begin,end in [(0,128),(128,133)]:
        test.append(ordinary_solve(w,inj[begin:end],np.full(end-begin,-1),1e-12))
    test_seconds=time.monotonic()-start
    # Full 120-row coarse panel, including every removed equation.
    start=time.monotonic();coarse=[]
    for unit in range(-1,32):
        coarse.append(ordinary_solve(w,inj[:120],np.full(120,unit),.001))
    panel_seconds=time.monotonic()-start
    ids=np.repeat(np.arange(120),32);units=np.tile(np.arange(32),120)
    base=coarse[0][0][ids];edit=np.stack([coarse[j+1][0][i] for i,j in zip(ids,units)])
    interval_inj=iv.affine_point(dummy[:120],a['input.weight'],a['input.bias'])
    interval_inj=tuple(endpoint[ids] for endpoint in interval_inj)
    winners=np.zeros(3840,dtype=np.int64);runners=np.ones(3840,dtype=np.int64)
    start=time.monotonic();last=None
    for _ in range(5):
        last=iv.effect_interval(base,edit,w,interval_inj,units,v,winners,runners,q)
    certificate_seconds=(time.monotonic()-start)/5
    start=time.monotonic()
    for _ in range(10):
        mapping(base,w,inj[:120][ids],np.full(3840,-1))
        mapping(edit,w,inj[:120][ids],units)
    paired_update_seconds=(time.monotonic()-start)/10
    start=time.monotonic()
    iv.logits_interval(test[0][0],w,iv.affine_point(dummy[:128],a['input.weight'],a['input.bias']),v,a['readout.bias'],q)
    full_certificate_seconds=time.monotonic()-start
    # Conservative refinement projection: all 3,840 pairs use all 129 checks.
    projected=2*(training_seconds*611680/2390 + test_seconds*7188/133 + panel_seconds*4
        + certificate_seconds*129*4 + paired_update_seconds*128*4 + full_certificate_seconds*60)
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['intervals.py','check_intervals.py','model.py','dummy_timing.py']}
    receipt=dict(status='passed' if projected<1200 else 'over_budget',scientific_model_outputs_collected=False,
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,code_sha256=hashes,
        training_shapes=[128,111],training_updates=20,training_examples=2390,test_shapes=[128,5],
        panel_rows=120,coarse_conditions=33,pair_batch=3840,
        training_seconds=training_seconds,test_seconds=test_seconds,panel_seconds=panel_seconds,
        certificate_3840_seconds=certificate_seconds,paired_update_3840_seconds=paired_update_seconds,
        full_certificate_128_seconds=full_certificate_seconds,
        projected_seconds_2x_with_all_pairs_at_cap=projected,
        elapsed_seconds=time.monotonic()-started,q_upper=q,
        interval_endpoints_finite=bool(np.isfinite(last['lower']).all() and np.isfinite(last['upper']).all()))
    np.savez_compressed(args.output/'dummy-model.npz',**a)
    np.savez_compressed(args.output/'dummy-pairs.npz',base=base,edit=edit,units=units,**last)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    assert projected<1200,'The fixed study scope does not fit the dummy projection.'


if __name__=='__main__':main()

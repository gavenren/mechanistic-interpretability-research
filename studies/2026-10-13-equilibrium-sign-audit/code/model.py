"""Small contractive ReLU model and ordinary floating-point solver."""
import numpy as np
import torch


class Model(torch.nn.Module):
    def __init__(self,seed):
        super().__init__()
        torch.manual_seed(seed)
        self.input=torch.nn.Linear(64,32,dtype=torch.float64)
        self.recurrent=torch.nn.Linear(32,32,bias=False,dtype=torch.float64)
        self.readout=torch.nn.Linear(32,10,dtype=torch.float64)
        self.project()
        assert sum(p.numel() for p in self.parameters())==3434

    def project(self):
        with torch.no_grad():
            w=self.recurrent.weight
            w.div_(torch.clamp(w.abs().sum(1,keepdim=True)/.875,min=1.))

    def forward(self,x):
        injection=self.input(x)
        state=torch.zeros(len(x),32,dtype=torch.float64)
        for _ in range(32):state=torch.relu(self.recurrent(state)+injection)
        return self.readout(state)

    def arrays(self):
        return {name:p.detach().numpy().copy() for name,p in self.named_parameters()}


def optimizer(model):
    return torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
        weight_decay=0,amsgrad=False,foreach=False,fused=False,maximize=False,
        capturable=False,differentiable=False)


def mapping(state,weight,injection,removed):
    value=np.maximum(state@weight.T+injection,0.)
    idx=np.flatnonzero(removed>=0)
    value[idx,removed[idx]]=0.
    assert np.isfinite(value).all()
    return value


def ordinary_solve(weight,injection,removed,tolerance,cap=512,callback=None):
    """Return first candidate passing its residual check, per image.

    The final residual evaluation is counted as a map evaluation, not an update.
    These residuals are ordinary float64 values, not certificates.
    """
    n=len(injection);state=np.zeros_like(injection)
    counts=np.zeros(n,dtype=np.int64);residual=np.full(n,np.inf)
    remaining=np.arange(n);calls=0;examples=0
    for updates in range(cap+1):
        if callback:callback()
        future=mapping(state[remaining],weight,injection[remaining],removed[remaining])
        calls+=1;examples+=len(remaining)
        errors=np.max(abs(future-state[remaining]),axis=1)
        residual[remaining]=errors
        keep=errors>tolerance
        if not keep.any():break
        if updates==cap:raise RuntimeError('Ordinary fixed-point cap reached.')
        idx=remaining[keep];state[idx]=future[keep];counts[idx]+=1
        remaining=idx
    return state,residual,counts,calls,examples

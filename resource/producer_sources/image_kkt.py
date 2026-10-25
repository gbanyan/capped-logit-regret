"""Continuous box-normal-cone posterior,not the natural image posterior."""
import numpy as np
import oracles as O


def construct(raw,cap,kappa):
    assert 0<=kappa<1 and cap>0
    raw=np.asarray(raw,dtype=np.float64);assert np.isfinite(raw).all()
    z=np.clip(raw,-cap,cap);q=np.exp(O.logs(z))
    logZ=np.log(np.exp(z-z.max(1,keepdims=True)).sum(1,keepdims=True))+z.max(1,keepdims=True)
    qlow=np.exp(-cap-logZ)
    a=np.maximum(0,raw-cap);b=np.maximum(0,-cap-raw)
    A=a.sum(1,keepdims=True);B=b.sum(1,keepdims=True);m=b.max(1,keepdims=True)
    positive=kappa*qlow*(B/(1+m))*(a/(1+A))
    negative=kappa*qlow*(A/(1+A))*(b/(1+m))
    p=q+positive-negative
    assert p.min()>0 and abs(p.sum(1)-1).max()<1e-12
    return dict(p=p,z=z,q=q,positive=positive,negative=negative,
        upper=(a>0),lower=(b>0),FW=O.fw_gap(p,z,'box',cap))

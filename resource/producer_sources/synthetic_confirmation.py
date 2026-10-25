"""Known-posterior neural control. All selection is development-only."""
import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import oracles as O


def write(p, d):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n'); tmp.replace(p)


def data(n,beta,seed,size,mode='realizable',noise=.05,ratio_override=None,shape=1.):
    rng=np.random.default_rng(seed)
    family=seed//100
    perm=np.random.default_rng(np.random.SeedSequence([family,19031])).permutation(n)
    if mode=='gaussian':
        assert n==10
        means=np.zeros((n,2,10))
        for label in range(n):
            group=label//2;sign=2*(label%2)-1
            means[label,:,group]=6.
            means[label,:,5+group]=sign*.5
            means[label,0,5+(group+1)%5]=.35
            means[label,1,5+(group+1)%5]=-.35
        labels=rng.integers(n,size=size);components=rng.integers(2,size=size)
        v=means[labels,components]+rng.normal(size=(size,10))
        ld=-.5*((v[:,None,None,:]-means[None,:,:,:])**2).sum(-1)
        largest=ld.max(2,keepdims=True)
        logdensity=largest[:,:,0]+np.log(np.exp(ld-largest).sum(2))
        pp=np.exp(O.logs(logdensity));p=np.empty_like(pp);p[:,perm]=pp
        x=np.column_stack([v/6,rng.uniform(-1,1,size=size)]).astype(np.float32)
        assert p.min()>0 and abs(p.sum(1)-1).max()<1e-12
        return x,p,dict(ratio=(n-2)*(1-beta)/(2*beta),beta=beta,
            permutation=perm.tolist(),means=means.tolist(),law='observable true marginalized Gaussian mixture')
    c=rng.integers(n,size=size); u=rng.uniform(size=size)
    v=np.eye(n)[c]+rng.uniform(-noise,noise,(size,n))
    assert np.array_equal(v.argmax(1),c)
    pair=np.stack([perm[c],perm[(c+1)%n]],axis=1)
    t=np.sign(2*u-1)*abs(2*u-1)**shape
    if n==2:
        ratio=ratio_override or 3.; beta=0.
        gamma=.98*t
        p=np.zeros((size,n)); p[np.arange(size),pair[:,0]]=(1+gamma)/2
        p[np.arange(size),pair[:,1]]=(1-gamma)/2
    else:
        ratio=(n-2)*(1-beta)/(2*beta)
        assert ratio>1
        if mode=='realizable':
            s=.95*beta; gamma=.9*beta*t
            low=1/(2*ratio+n-2); high=ratio*low
            p=np.full((size,n),low-s/(n-2))
            p[np.arange(size),pair[:,0]]=high+(s+gamma)/2
            p[np.arange(size),pair[:,1]]=high+(s-gamma)/2
        elif mode=='mixed':
            tail=.1*beta
            gamma=(1-tail-2*tail/(n-2)-.001)*t
            p=np.full((size,n),tail/(n-2))
            p[np.arange(size),pair[:,0]]=(1-tail+gamma)/2
            p[np.arange(size),pair[:,1]]=(1-tail-gamma)/2
        else: raise ValueError(mode)
    assert p.min()>0 and abs(p.sum(1)-1).max()<1e-12
    return np.column_stack([v,2*u-1]).astype(np.float32),p,dict(ratio=ratio,beta=beta,permutation=perm.tolist())


class Model(nn.Module):
    def __init__(self,n,width,depth,head,ratio):
        super().__init__(); layers=[]; before=n+1
        for _ in range(depth):
            layers.extend([nn.Linear(before,width),nn.ReLU()]); before=width
        layers.append(nn.Linear(before,n)); self.net=nn.Sequential(*layers)
        self.head=head; self.cap=math.log(ratio)/2; self.radius=math.sqrt(2)*self.cap

    def forward(self,x,raw=False):
        z=self.net(x)
        if raw or self.head=='raw': return z
        if self.head=='box': return z.clamp(-self.cap,self.cap)
        if self.head=='tanh': return self.cap*torch.tanh(z/self.cap)
        if self.head=='ball': return z*(self.radius/z.norm(dim=1,keepdim=True).clamp_min(1e-20)).clamp(max=1)
        raise ValueError(self.head)


def oracle(p,head,ratio):
    if head=='raw':
        z=np.log(p);z-=z.mean(1,keepdims=True)
        return dict(z=z,q=p.copy(),CE=O.loss(p,z),gap=np.zeros(len(p)))
    return O.ball(p,math.sqrt(2)*math.log(ratio)/2) if head=='ball' else O.box(p,ratio)


def metrics(p,z,oo,head,beta,raw_scores=None,ratio_override=None):
    adjustment=0.
    if head!='raw':
        ratio=(ratio_override or 3.) if p.shape[1]==2 else (p.shape[1]-2)*(1-beta)/(2*beta)
        cap=math.log(ratio)/2
        feasible=O.project_ball(z,math.sqrt(2)*cap) if head=='ball' else np.clip(z,-cap,cap)
        adjustment=float(abs(feasible-z).max())
        assert adjustment<2e-6,'More than floating point head feasibility error'
        z=feasible
    d=O.loss(p,z)-oo['CE']; upper=np.maximum(0,d+oo['gap']+1e-12)
    assert d.min()>-2e-9
    pred=z.argmax(1);last=p.shape[1]-1-z[:,::-1].argmax(1)
    regret=p.max(1)-p[np.arange(len(p)),pred]
    cert=O.prediction_certificate(p,z,oo)
    assert ((pred==p.argmax(1))|~cert['certified']).all()
    qt=oo['q']; top=np.partition(qt,-2,axis=1)
    tied=abs(top[:,-1]-top[:,-2])<1e-10
    zz=np.partition(z,-2,axis=1)
    if head in ('box','tanh') and beta>0:
        ratio=(p.shape[1]-2)*(1-beta)/(2*beta)
        slope=1+beta+2/(ratio-1)
        assert (regret-np.minimum(1,beta+slope*np.sqrt(2*upper))).max()<1e-8
    result=dict(mean_D=float(d.mean()),D_quantiles=np.quantile(d,[0,.1,.5,.9,1]).tolist(),
        near_1e3=float((upper<=1e-3).mean()),regret=float(regret.mean()),
        regret_last_index=float((p.max(1)-p[np.arange(len(p)),last]).mean()),
        Bayes_disagreement=float((pred!=p.argmax(1)).mean()),
        exact_score_tie=float((zz[:,-1]==zz[:,-2]).mean()),
        near_score_tie_1e7=float((abs(zz[:,-1]-zz[:,-2])<=1e-7).mean()),
        tied_oracle_mass=float(tied.mean()),certified_correct_mass=float(cert['certified'].mean()),
        oracle_max_FW_gap=float(oo['gap'].max()),FP32_feasibility_adjustment=adjustment)
    if 'lower_probability' in oo:
        threshold=ratio*oo['lower_probability']
        result['analytic_top_clip_mass_sensitivity']={str(tol):float(((p>=threshold-tol).sum(1)>=2).mean())
                                                     for tol in (0.,1e-12,1e-10)}
    # Conditional reference uses the ACTUAL selected near set,not an unconditional
    # population formula after outcome-dependent stratum selection.
    near=upper<=1e-3
    optimum_top=oo['q'].max(1,keepdims=True)
    optimal_ties=abs(oo['q']-optimum_top)<=1e-12
    random_tie_regret=p.max(1)-(p*optimal_ties).sum(1)/optimal_ties.sum(1)
    result['oracle_random_tie_regret']=float(random_tie_regret.mean())
    result['oracle_random_tie_regret_on_N']=float(random_tie_regret[near].mean()) if near.any() else None
    if raw_scores is not None:
        if head in ('tanh','ball'):
            ideal_pred=raw_scores.argmax(1)  # mathematically strictly monotone/radial positive map
            result['ideal_monotone_Bayes_disagreement']=float((ideal_pred!=p.argmax(1)).mean())
            result['ideal_monotone_regret']=float((p.max(1)-p[np.arange(len(p)),ideal_pred]).mean())
            result['rounded_vs_monotone_decision_fraction']=float((pred!=ideal_pred).mean())
        if head=='tanh':
            z64=cap*np.tanh(raw_scores/cap)
            t64=np.partition(z64,-2,axis=1)
            result['FP64_head_exact_tie']=float((t64[:,-1]==t64[:,-2]).mean())
            result['FP64_head_D']=float((O.loss(p,z64)-oo['CE']).mean())
        if head in ('box','tanh'):
            result['raw_coordinate_saturation']=float((abs(raw_scores)>=cap).mean())
            result['wrong_side_saturation']=float((((raw_scores<=-cap)&(oo['z']>-cap+1e-6))|
                ((raw_scores>=cap)&(oo['z']<cap-1e-6))).mean())
    return result


def evaluate(model,x,raw=False):
    model.eval()
    with torch.no_grad(): return torch.cat([model(xx,raw=raw) for xx in x.split(512)]).cpu().numpy()


def train(job,out):
    out.mkdir(exist_ok=False)
    n,beta,seed=job['n'],job['beta'],job['seed']
    options=dict(mode=job.get('mode','realizable'),noise=job.get('noise',.05),ratio_override=job.get('ratio'),shape=job.get('shape',1.))
    x,p,meta=data(n,beta,seed*100,job['training'],**options)
    xv,pv,vm=data(n,beta,seed*100+1,job['validation'],**options)
    assert meta['permutation']==vm['permutation']
    ratio=meta['ratio']; head=job['head']
    oo=oracle(p,head,ratio); ov=oracle(pv,head,ratio)
    target=oo['z'].copy()
    if head=='tanh':
        cap=math.log(ratio)/2; epsilon=min(.01,1e-4/(2*cap))
        target=cap*np.arctanh(np.clip((1-epsilon)*target/cap,-1+epsilon,1-epsilon))
    torch.manual_seed(seed)
    model=Model(n,job['width'],job['depth'],head,ratio).cuda()
    if job.get('saturated_init'):
        generator=torch.Generator(device='cuda').manual_seed(seed+773)
        signs=torch.where(torch.rand(n,generator=generator,device='cuda')<.5,-1.,1.)
        with torch.no_grad():model.net[-1].bias.copy_(2*model.cap*signs)
    initialization=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    dtype=torch.float64 if job.get('dtype')=='float64' else torch.float32
    model=model.to(dtype=dtype)
    xt=torch.tensor(x,dtype=dtype,device='cuda'); pt=torch.tensor(p,dtype=dtype,device='cuda')
    vt=torch.tensor(xv,dtype=dtype,device='cuda'); targets=torch.tensor(target,dtype=dtype,device='cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,job['epochs'],eta_min=.00002)
    shuffle=torch.Generator(device='cuda').manual_seed(seed+11)
    start=time.monotonic();history=[]
    for epoch in range(job['epochs']+1):
        if epoch%25==0 or epoch==job['epochs']:
            z=evaluate(model,vt).astype(np.float64)
            raw_scores=evaluate(model,vt,raw=True).astype(np.float64)
            try:report=metrics(pv,z,ov,head,meta['beta'],raw_scores,ratio)
            except Exception:
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'failure_model.pt')
                np.savez_compressed(out/'failure_validation.npz',x=xv,p=pv,scores=z,raw_scores=raw_scores,oracle_scores=ov['z'],oracle_gap=ov['gap'])
                write(out/'failure_context.json',dict(job=job,epoch=epoch,history=history,traceback=traceback.format_exc()))
                raise
            history.append(dict(epoch=epoch,validation=report,seconds=time.monotonic()-start))
            write(out/'progress.json',dict(job=job,latest=history[-1]))
            print(json.dumps(dict(job=out.name,epoch=epoch,mean_D=report['mean_D'],near=report['near_1e3'])),flush=True)
        if epoch==job['epochs']:break
        model.train()
        order=torch.randperm(len(x),generator=shuffle,device='cuda')
        for ids in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            if job['objective']=='CE': objective=-(pt[ids]*F.log_softmax(model(xt[ids]),dim=1)).sum(1).mean()
            else: objective=F.mse_loss(model(xt[ids],raw=True),targets[ids])
            objective.backward();optimizer.step()
        scheduler.step()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
    training_scores=evaluate(model,xt).astype(np.float64)
    training_raw=evaluate(model,xt,raw=True).astype(np.float64)
    np.savez_compressed(out/'validation.npz',x=xv,p=pv,scores=z,raw_scores=raw_scores,oracle_scores=ov['z'],oracle_gap=ov['gap'])
    fresh=Model(n,job['width'],job['depth'],head,ratio).cuda().to(dtype=dtype)
    fresh.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
    assert np.array_equal(evaluate(fresh,vt).astype(np.float64),z)
    write(out/'complete.json',dict(job=job,meta=meta,initialization_sha256=initialization,
        history=history,training=metrics(p,training_scores,oo,head,meta['beta'],training_raw,ratio),
        validation=report,replay_exact=True,seconds=time.monotonic()-start,
        model_sha256=hashlib.sha256((out/'model.pt').read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(x.tobytes()+p.tobytes()+xv.tobytes()+pv.tobytes()).hexdigest(),
        torch=torch.__version__,CUDA=torch.version.cuda))


def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    queue=json.loads(args.queue.read_text())
    failures=[]
    for j in queue['jobs']:
        try:train(j,args.output/j['name'])
        except Exception:
            failures.append(j['name'])
            path=args.output/j['name'];path.mkdir(exist_ok=True)
            write(path/'failure.json',dict(job=j,traceback=traceback.format_exc()))
            print('FAILED '+j['name'],flush=True)
    write(args.output/'queue_complete.json',dict(jobs=len(queue['jobs']),failed=failures,all_successful=not failures))
    if failures:raise RuntimeError('Preserved failing jobs: '+str(failures))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())

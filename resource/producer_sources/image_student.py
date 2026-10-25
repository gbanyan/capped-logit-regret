"""Known-posterior frozen-representation neural students;no image test access."""
import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import synthetic_confirmation as S


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(job,data_root):
    feature_phase=job.get('feature_phase','image_features_v1')
    folder=data_root/'output'/feature_phase/job['architecture']
    complete=json.loads((folder/'complete.json').read_text())
    assert sha(folder/'features.npz')==job['features_sha256']==complete['features_sha256']
    assert not complete['official_test_evaluated']
    a=np.load(folder/'features.npz',allow_pickle=False);n=a['logits'].shape[1];u=a['u'];logits=a['logits']
    beta=job['beta'];tail=.1*beta;low=tail/(n-2);gamma=(1-tail-2*low-.001)*(2*u-1)
    first=(1-tail+gamma)/2;second=(1-tail-gamma)/2
    w=np.exp(S.O.logs(logits.astype(np.float64)))
    p=low+(first-low)[:,None]*w+(second-low)[:,None]*np.roll(w,1,axis=1)
    permutation=np.random.default_rng(np.random.SeedSequence([job['seed'],19031])).permutation(n)
    pp=np.empty_like(p);pp[:,permutation]=p;p=pp
    assert p.min()>0 and abs(p.sum(1)-1).max()<1e-12
    x=np.column_stack([(logits-a['feature_mean'])/a['feature_std'],2*u-1]).astype(np.float32)
    training=a['train_indices'].copy();validation=a['validation_indices'].copy()
    if job.get('training_subset'):
        training=np.random.default_rng(job['seed']+29).choice(training,job['training_subset'],replace=False);training.sort()
    identity=dict(teacher_architecture=job['architecture'],features_sha256=job['features_sha256'],feature_phase=feature_phase,
        n=n,beta=beta,ratio=(n-2)*(1-beta)/(2*beta),permutation=permutation.tolist(),
        training_examples=len(training),validation_examples=len(validation),
        training_indices_sha256=hashlib.sha256(training.tobytes()).hexdigest(),
        validation_indices_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
        official_test_evaluated=False)
    return x,p,training,validation,identity


def train(job,out,args):
    out.mkdir(exist_ok=False);start=time.monotonic()
    x,p,training,validation,meta=load(job,args.data_root)
    x,p,xv,pv=x[training],p[training],x[validation],p[validation]
    n,beta,ratio,head=meta['n'],meta['beta'],meta['ratio'],job['head']
    oo=S.oracle(p,head,ratio);ov=S.oracle(pv,head,ratio);target=oo['z'].copy()
    if head=='tanh':
        cap=math.log(ratio)/2;epsilon=min(.01,1e-4/(2*cap))
        target=cap*np.arctanh(np.clip((1-epsilon)*target/cap,-1+epsilon,1-epsilon))
    device=job.get('device','cuda');torch.manual_seed(job['seed'])
    model=S.Model(n,job['width'],job['depth'],head,ratio).to(device)
    initialization=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    xt=torch.tensor(x,device=device);pt=torch.tensor(p,dtype=torch.float32,device=device)
    vt=torch.tensor(xv,device=device);targets=torch.tensor(target,dtype=torch.float32,device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,job['epochs'],eta_min=.00002)
    shuffle=torch.Generator(device=device).manual_seed(job['seed']+11);history=[]
    S.write(out/'input.json',meta)
    for epoch in range(job['epochs']+1):
        if epoch%25==0 or epoch==job['epochs']:
            z=S.evaluate(model,vt).astype(np.float64);raw=S.evaluate(model,vt,raw=True).astype(np.float64)
            try:report=S.metrics(pv,z,ov,head,beta,raw,ratio)
            except Exception:
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'failure_model.pt')
                np.savez_compressed(out/'failure_validation.npz',x=xv,p=pv,scores=z,raw_scores=raw,
                    oracle_scores=ov['z'],oracle_gap=ov['gap'])
                S.write(out/'failure_context.json',dict(epoch=epoch,history=history,job=job,traceback=traceback.format_exc()));raise
            history.append(dict(epoch=epoch,validation=report,seconds=time.monotonic()-start))
            S.write(out/'progress.json',dict(job=job,latest=history[-1]))
            print(json.dumps(dict(job=job['name'],epoch=epoch,mean_D=report['mean_D'],near=report['near_1e3'],seconds=history[-1]['seconds'])),flush=True)
        if epoch==job['epochs']:break
        model.train();order=torch.randperm(len(x),generator=shuffle,device=device)
        for ids in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            if job['objective']=='CE':objective=-(pt[ids]*F.log_softmax(model(xt[ids]),dim=1)).sum(1).mean()
            else:objective=F.mse_loss(model(xt[ids],raw=True),targets[ids])
            objective.backward();optimizer.step()
        scheduler.step()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
    training_z=S.evaluate(model,xt).astype(np.float64);training_raw=S.evaluate(model,xt,raw=True).astype(np.float64)
    np.savez_compressed(out/'validation.npz',x=xv,p=pv,scores=z,raw_scores=raw,oracle_scores=ov['z'],oracle_gap=ov['gap'],indices=validation)
    fresh=S.Model(n,job['width'],job['depth'],head,ratio).to(device)
    fresh.load_state_dict(torch.load(out/'model.pt',weights_only=True,map_location='cpu'))
    assert np.array_equal(S.evaluate(fresh,vt).astype(np.float64),z)
    S.write(out/'complete.json',dict(job=job,meta=meta,history=history,validation=report,
        training=S.metrics(p,training_z,oo,head,beta,training_raw,ratio),initialization_sha256=initialization,
        model_sha256=sha(out/'model.pt'),replay_exact=True,seconds=time.monotonic()-start,
        torch=torch.__version__,device=device,official_test_evaluated=False,
        parameter_count=sum(v.numel() for v in model.parameters())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cifar',type=Path);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    failures=[]
    for j in json.loads(args.queue.read_text())['jobs']:
        try:train(j,args.output/j['name'],args)
        except Exception:
            failures.append(j['name']);p=args.output/j['name'];p.mkdir(exist_ok=True)
            S.write(p/'failure.json',dict(job=j,traceback=traceback.format_exc()));print('FAILED '+j['name'],flush=True)
    S.write(args.output/'queue_complete.json',dict(failed=failures,all_successful=not failures))
    if failures:raise RuntimeError('Preserved failed image student arms: '+str(failures))

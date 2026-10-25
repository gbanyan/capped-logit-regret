"""Frozen teacher body with a random normalized final affine;actual pixel gates."""
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
import affine_features as A
import image_pixel as P
import synthetic_confirmation as S


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_head(raw,head,ratio):
    cap=math.log(ratio)/2
    if head=='raw':return raw
    if head=='box':return raw.clamp(-cap,cap)
    if head=='tanh':return cap*torch.tanh(raw/cap)
    if head=='ball':return raw*(math.sqrt(2)*cap/raw.norm(dim=1,keepdim=True).clamp_min(1e-20)).clamp(max=1)
    raise ValueError(head)


def fold_head(weight,bias,mean,std):
    w=weight.astype(np.float64)/std.astype(np.float64)[None,:]
    b=bias.astype(np.float64)-w@mean.astype(np.float64)
    assert np.isfinite(w).all() and np.isfinite(b).all()
    return w.astype(np.float32),b.astype(np.float32)


def predict_head(head,features,kind,ratio,raw=False):
    head.eval();result=[]
    with torch.no_grad():
        for chunk in features.split(128):
            r=head(chunk);result.append((r if raw else apply_head(r,kind,ratio)).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def train_epoch(head,features,target,kind,ratio,optimizer,generator):
    head.train();order=torch.randperm(len(features),generator=generator,device=features.device);steps=0
    for ids in order.split(128):
        optimizer.zero_grad(set_to_none=True);z=apply_head(head(features[ids]),kind,ratio)
        objective=-(target[ids]*F.log_softmax(z,dim=1)).sum(1).mean()
        assert bool(torch.isfinite(objective))
        objective.backward()
        assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in head.parameters())
        optimizer.step();steps+=1
    return steps


def run(job,args):
    out=args.output/job['name'];out.mkdir(exist_ok=False);start=time.monotonic()
    assert job['law']=='image_kkt' and job['targets']=='soft' and job['beta']==.3 and job['kappa']==.95
    assert job['epochs']==100 and job['affine_arm'] in ('primary','secondary')
    assert job['initialization']=='random_last_affine' and job['freeze_body']
    cache=args.data_root/'output'/job['affine_feature_phase']/job['architecture']
    receipt=json.loads((cache/'complete.json').read_text())
    assert sha(cache/'features.npz')==job['affine_features_sha256']==receipt['features_sha256']
    assert sha(cache/'complete.json')==job['affine_cache_complete_sha256'] and not (cache/'failure.json').exists()
    assert receipt['source_feature_sha256']==job['features_sha256'] and receipt['source_teacher_sha256']==job['teacher_sha256']
    with np.load(cache/'features.npz',allow_pickle=False) as packed:a={key:packed[key] for key in packed.files}
    assert all(np.isfinite(value).all() for value in a.values())
    x,u,p,training,validation,meta,clean=P.prepare(job,args)
    perm=np.asarray(meta['permutation'])
    for split,ids in [('training',training),('validation',validation)]:
        assert np.array_equal(a[split+'_indices'],ids)
        assert np.array_equal(p[ids][:,perm],a['p_'+split])
    model=P.Pixel(job,meta['n'],meta['ratio']).cuda().eval()
    assert sha(clean/'centered_teacher.pt')==job['teacher_sha256']
    model.base.load_state_dict(torch.load(clean/'centered_teacher.pt',map_location='cpu',weights_only=True))
    name,native=A.final_affine(model);exclude=(name+'.weight',name+'.bias')
    body=A.state_sha(model.state_dict(),exclude);assert body==receipt['body_state_sha256']
    for parameter in model.parameters():parameter.requires_grad_(False)
    d=native.in_features;n=meta['n'];ratio=meta['ratio'];beta=job['beta'];kind=job['head']
    mean=a['feature_mean'];std=a['feature_std'];assert mean.shape==std.shape==(d,) and np.all(std>=.1)
    xb=torch.tensor(((a['training_features'].astype(np.float64)-mean)/std).astype(np.float32),device='cuda')
    xv=torch.tensor(((a['validation_features'].astype(np.float64)-mean)/std).astype(np.float32),device='cuda')
    target=torch.tensor(p[training],dtype=torch.float32,device='cuda')
    torch.manual_seed(job['seed']);head=nn.Linear(d,n,device='cuda')
    initial=A.state_sha(head.state_dict())
    expected_lr=(.1 if job['architecture']=='resnet18' else .001)/(10 if job['affine_arm']=='secondary' else 1)
    assert job['learning_rate']==expected_lr
    optimizer=torch.optim.SGD(head.parameters(),lr=expected_lr,momentum=.9,weight_decay=0) if job['architecture']=='resnet18' else torch.optim.Adam(head.parameters(),lr=expected_lr,weight_decay=0)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,eta_min=expected_lr*.001)
    generator=torch.Generator(device='cuda').manual_seed(job['seed']+11);proxy=[];steps=0
    oracle_b=S.oracle(p[training],kind,ratio);oracle_v=S.oracle(p[validation],kind,ratio)
    S.write(out/'input.json',meta)
    for epoch in range(101):
        if epoch%5==0:
            vr=predict_head(head,xv,kind,ratio,raw=True);vz=predict_head(head,xv,kind,ratio)
            report=S.metrics(p[validation],vz,oracle_v,kind,beta,vr,ratio)
            proxy.append(dict(epoch=epoch,cached_validation_proxy=report,seconds=time.monotonic()-start))
            S.write(out/'progress.json',dict(job=job,latest=proxy[-1],gate_scope='Cached proxy only;final gate requires whole-pixel forward'))
            print(json.dumps(dict(job=job['name'],epoch=epoch,cached_D=report['mean_D'],cached_near=report['near_1e3'])),flush=True)
        if epoch==100:break
        steps+=train_epoch(head,xb,target,kind,ratio,optimizer,generator)
        schedule.step()
    torch.save({key:value.detach().cpu() for key,value in head.state_dict().items()},out/'normalized_head.pt')
    w,b=fold_head(head.weight.detach().cpu().numpy(),head.bias.detach().cpu().numpy(),mean,std)
    with torch.no_grad():native.weight.copy_(torch.tensor(w,device='cuda'));native.bias.copy_(torch.tensor(b,device='cuda'))
    assert A.state_sha(model.state_dict(),exclude)==body
    torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},out/'model.pt')
    reports={};deltas={}
    for split,ids,features,oracle in [('training',training,xb,oracle_b),('validation',validation,xv,oracle_v)]:
        ix=torch.tensor(ids,device='cuda');z=P.predictions(model,x,u,ix);raw=P.predictions(model,x,u,ix,raw=True)
        cr=predict_head(head,features,kind,ratio,raw=True);cz=predict_head(head,features,kind,ratio)
        reports[split]=S.metrics(p[ids],z,oracle,kind,beta,raw,ratio)
        cache_report=S.metrics(p[ids],cz,oracle,kind,beta,cr,ratio)
        deltas[split]=dict(raw_max_difference=float(abs(cr-raw).max()),score_max_difference=float(abs(cz-z).max()),
            CE_signed_difference=float(S.O.loss(p[ids],z).mean()-S.O.loss(p[ids],cz).mean()),
            mean_D_signed_difference=reports[split]['mean_D']-cache_report['mean_D'],
            decision_disagreement=float((cz.argmax(1)!=z.argmax(1)).mean()),cached_metrics=cache_report,
            deployed_metrics=reports[split],bitwise_equal=bool(np.array_equal(cr,raw) and np.array_equal(cz,z)))
        np.savez_compressed(out/(split+'.npz'),p=p[ids],scores=z,raw_scores=raw,indices=ids,
            oracle_scores=oracle['z'],oracle_gap=oracle['gap'])
        fresh=P.Pixel(job,n,ratio).cuda().eval();fresh.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
        assert np.array_equal(P.predictions(fresh,x,u,ix),z)
        assert np.array_equal(P.predictions(fresh,x,u,ix,raw=True),raw)
        assert A.state_sha(fresh.state_dict(),exclude)==body
        del fresh
    assert A.state_sha(model.state_dict(),exclude)==body
    S.write(out/'complete.json',dict(job=job,meta=meta,validation=reports['validation'],training=reports['training'],
        history=[dict(epoch=100,validation=reports['validation'],seconds=time.monotonic()-start)],cached_proxy_history=proxy,
        initialization_sha256=initial,normalized_initialization_sha256=initial,
        model_sha256=sha(out/'model.pt'),normalized_head_sha256=sha(out/'normalized_head.pt'),
        training_arrays_sha256=sha(out/'training.npz'),validation_arrays_sha256=sha(out/'validation.npz'),
        body_state_sha256=body,final_affine_name=name,trainable_state_keys=list(exclude),
        final_affine_only_optimized=True,body_parameters_and_buffers_unchanged=True,body_eval_mode_throughout=True,
        feature_normalizer_fit='studentB only',folding='FP64 arithmetic then one FP32 cast',
        cached_vs_deployed=deltas,replay_exact=True,raw_replay_exact=True,
        trained_seed=True,derived_witness=False,optimizer_steps=steps,
        parameter_count=sum(v.numel() for v in model.parameters()),trainable_parameter_count=sum(v.numel() for v in head.parameters()),
        seconds=time.monotonic()-start,torch=torch.__version__,CUDA=torch.version.cuda,
        source_sha256=sha(Path(__file__)),official_test_evaluated=False,
        scope='Fixed learned teacher representation,random last affine;actual deployed pixel gates,not random end-to-end learning'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--cifar',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    jobs=json.loads(args.queue.read_text())['jobs'];assert len(jobs)==60 and len({j['name'] for j in jobs})==60
    failed=[]
    for job in jobs:
        try:run(job,args)
        except Exception:
            failed.append(job['name']);folder=args.output/job['name'];folder.mkdir(exist_ok=True)
            S.write(folder/'failure.json',dict(job=job,traceback=traceback.format_exc()))
    S.write(args.output/'queue_complete.json',dict(jobs=len(jobs),failed=failed,all_successful=not failed))
    if failed:raise RuntimeError('Preserved affine student failures:'+str(failed))

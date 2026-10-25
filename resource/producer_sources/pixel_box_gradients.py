"""All69 original pixelbox checkpoints,full-B eval gradients,zero updates."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import image_pixel as P
import synthetic_confirmation as S


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def gradients(model,x,u,ids,p,raw_reference,labels=None,scores_reference=None):
    model.eval();model.zero_grad(set_to_none=True);actual=[];virtual=[];raws=[]
    count=len(ids);value=0.
    for start in range(0,count,128):
        ix=ids[start:start+128];raw=model(x[ix],u[ix],raw=True);raw.retain_grad()
        assert bool(torch.isfinite(raw).all()),'Nonfinite raw scores require explicit diagnosis'
        assert np.array_equal(raw.detach().cpu().numpy().astype(np.float64),raw_reference[start:start+len(ix)])
        z=raw.clamp(-model.cap,model.cap);logq=F.log_softmax(z,dim=1)
        assert bool(torch.isfinite(logq).all())
        if scores_reference is not None:
            assert np.array_equal(z.detach().cpu().numpy().astype(np.float64),scores_reference[start:start+len(ix)]),'Diagnostic head differs from deployed head'
        if labels is None:
            target=torch.tensor(p[start:start+len(ix)],dtype=torch.float32,device=x.device)
            per=-(target*logq).sum(1)
        else:
            yy=torch.tensor(labels[start:start+len(ix)],device=x.device)
            target=F.one_hot(yy,num_classes=raw.shape[1]).float()
            per=F.nll_loss(logq,yy,reduction='none')
        vg=logq.exp()*target.sum(1,keepdim=True)-target
        loss=per.sum()/count;value+=float(loss.detach());loss.backward()
        ag=raw.grad.detach()*count
        assert bool(torch.isfinite(vg).all()) and bool(torch.isfinite(ag).all())
        cap=torch.tensor(model.cap,dtype=raw.dtype,device=raw.device)
        strict=abs(raw.detach())>cap
        assert not torch.count_nonzero(ag[strict]),'Strict clamp must kill instantaneous raw gradient'
        actual.append(ag.cpu().numpy());virtual.append(vg.detach().cpu().numpy());raws.append(raw.detach().cpu().numpy())
    final=model.readout.net[-1] if model.readout is not None else model.base.output
    if not isinstance(final,nn.Linear):final=final[-1]
    assert isinstance(final,nn.Linear) and final.bias is not None
    for name,parameter in model.named_parameters():
        if parameter.grad is not None:assert bool(torch.isfinite(parameter.grad).all()),'Nonfinite parameter gradient:'+name
    report=dict(mean_eval_CE=value,
        full_parameter_gradient_linf=max(float(v.grad.abs().max()) for v in model.parameters() if v.grad is not None),
        output_weight_row_gradient_linf=final.weight.grad.detach().abs().amax(1).cpu().tolist(),
        output_bias_gradient=final.bias.grad.detach().cpu().tolist(),
        raw_forward_bitwise_matches_deployed=True,
        scores_forward_bitwise_matches_deployed=scores_reference is not None,all_parameter_gradients_finite=True)
    return report,np.concatenate(raws),np.concatenate(actual),np.concatenate(virtual)


def corrective_profile(raw,gradient,cap):
    upper=raw>cap;lower=raw<-cap
    directional=np.where(upper,gradient,np.where(lower,-gradient,0))
    positive=directional[directional>0]
    grid=[]
    for threshold in (0.,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4):
        selected=directional>threshold
        grid.append(dict(threshold=threshold,coordinates=int(selected.sum()),examples=int(selected.any(1).sum()),fraction=float(selected.mean())))
    return dict(by_threshold=grid,positive_direction_magnitude_quantiles=np.quantile(positive,[0,.1,.5,.9,.99,1]).tolist() if len(positive) else None)


def run(job,args):
    out=args.output/job['name'];out.mkdir(exist_ok=False)
    source=args.data_root/'output'/job['source_phase']/job['source_job']
    record=json.loads((source/'complete.json').read_text());j=record['job']
    assert j['head']=='box' and not record['official_test_evaluated']
    assert sha(source/'model.pt')==job['source_model_sha256']==record['model_sha256']
    assert sha(source/'complete.json')==job['source_complete_sha256']
    assert sha(source/'validation.npz')==job['source_validation_sha256']
    x,u,p,training,validation,meta,_=P.prepare(j,args);assert meta==record['meta']
    model=P.Pixel(j,meta['n'],meta['ratio']).cuda()
    model.load_state_dict(torch.load(source/'model.pt',map_location='cpu',weights_only=True));model.eval()
    before=state_sha(model)
    ti=torch.tensor(training,device='cuda');vi=torch.tensor(validation,device='cuda')
    saved=np.load(source/'validation.npz',allow_pickle=False)
    assert np.array_equal(saved['p'],p[validation]) and np.array_equal(saved['indices'],validation)
    assert np.array_equal(P.predictions(model,x,u,vi),saved['scores'])
    assert np.array_equal(P.predictions(model,x,u,vi,raw=True),saved['raw_scores'])
    raw=P.predictions(model,x,u,ti,raw=True);scores=P.predictions(model,x,u,ti)
    oo=S.oracle(p[training],'box',meta['ratio'])
    metrics=S.metrics(p[training],scores,oo,'box',j['beta'],raw,meta['ratio'])
    assert abs(metrics['mean_D']-record['training']['mean_D'])<1e-10
    assert abs(metrics['regret']-record['training']['regret'])<1e-10
    soft,raw32,actual,virtual=gradients(model,x,u,ti,p[training],raw,scores_reference=scores)
    cap32=np.float32(model.cap);upper=raw32>cap32;lower=raw32<-cap32;strict=upper|lower
    corrective=((upper&(virtual>1e-7))|(lower&(virtual<-1e-7)))
    q64=np.exp(S.O.logs(scores));p64=p[training]
    conditional64=q64*p64.sum(1,keepdims=True)-p64
    assert np.isfinite(conditional64).all()
    arrays=dict(indices=training,raw_scores=raw32,actual_soft_raw_gradient=actual,
        virtual_soft_raw_gradient=virtual,conditional_soft_score_gradient_FP64=conditional64,cap32=np.asarray(cap32))
    hard=None
    if j['targets']=='hard':
        assert sha(source/'sampled_labels.npz')==job['sampled_labels_sha256']
        sampled=np.load(source/'sampled_labels.npz',allow_pickle=False)
        assert np.array_equal(sampled['indices'],training)
        cumulative=p[training].cumsum(1);cumulative[:,-1]=1
        draw=np.random.default_rng(j['seed']+4141).uniform(size=len(training))
        labels=(draw[:,None]>cumulative).sum(1).astype(np.int64)
        assert np.array_equal(labels,sampled['labels'])
        hard,hr,ha,hv=gradients(model,x,u,ti,p[training],raw,labels=labels,scores_reference=scores)
        assert np.array_equal(hr,raw32)
        hard64=q64.copy();hard64[np.arange(len(labels)),labels]-=1
        arrays.update(sampled_labels=labels,actual_hard_raw_gradient=ha,virtual_hard_raw_gradient=hv,
            conditional_hard_score_gradient_FP64=hard64)
    assert state_sha(model)==before,'No parameter or BN-buffer updates are allowed'
    np.savez_compressed(out/'training_gradients.npz',**arrays)
    result=dict(job=job,source_training_job=j,source_validation=record['validation'],training=metrics,
        training_examples=len(training),validation_examples=len(validation),validation_replay_bitwise=True,
        meta=meta,soft_eval_gradient=soft,hard_eval_gradient=hard,
        strictly_clipped_coordinate_fraction=float(strict.mean()),
        strict_upper_coordinate_fraction=float(upper.mean()),strict_lower_coordinate_fraction=float(lower.mean()),
        all_train_strictly_clipped_coordinates=np.flatnonzero(strict.all(0)).tolist(),
        all_train_strict_upper_coordinates=np.flatnonzero(upper.all(0)).tolist(),
        all_train_strict_lower_coordinates=np.flatnonzero(lower.all(0)).tolist(),
        corrective_dead_coordinate_fraction=float(corrective.mean()),
        corrective_dead_examples=int(corrective.any(1).sum()),
        corrective_virtual_gradient_max=float(abs(virtual[corrective]).max()) if corrective.any() else 0.,
        actual_gradient_max_on_corrective_dead=float(abs(actual[corrective]).max()) if corrective.any() else 0.,
        machine_FP32_corrective_profile=corrective_profile(raw32,virtual,cap32),
        conditional_FP64_corrective_profile=corrective_profile(raw32,conditional64,cap32),
        machine_vs_conditional_virtual_gradient_max_difference=float(abs(virtual.astype(np.float64)-conditional64).max()),
        conditional_gradient_scope='FP64 conditional-loss derivative at SAME deployed FP32 scores,not FP64 network inference',
        original_per_model_gate=record['validation']['mean_D']<=1e-3 and record['validation']['near_1e3']>=.5,
        parameter_and_buffer_bytes_unchanged=True,checkpoint_state_sha256=before,
        arrays_sha256=sha(out/'training_gradients.npz'),source_sha256=sha(Path(__file__)),
        optimizer_steps=0,official_test_evaluated=False,trained_seed=False,
        torch=torch.__version__,CUDA=torch.version.cuda,
        scope='Frozen eval-mode full-B checkpoint gradients,not historical/stochastic training gradients or proof of a permanent trap')
    S.write(out/'complete.json',result)
    print(json.dumps(dict(job=job['name'],soft_gradient_linf=soft['full_parameter_gradient_linf'],
        corrective_dead_examples=result['corrective_dead_examples'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cifar',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    failures=[]
    jobs=json.loads(args.queue.read_text())['jobs'];assert len(jobs)==69 and len({j['name'] for j in jobs})==69
    for job in jobs:
        try:run(job,args)
        except Exception:
            failures.append(job['name']);out=args.output/job['name'];out.mkdir(exist_ok=True)
            S.write(out/'failure.json',dict(job=job,traceback=traceback.format_exc()))
    S.write(args.output/'queue_complete.json',dict(jobs=len(jobs),failed=failures,all_successful=not failures))
    if failures:raise RuntimeError('Preserved pixel gradient diagnostic failures: '+str(failures))

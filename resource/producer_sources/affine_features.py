"""Frozen FP32 native final-affine inputs on B/validation;no classifier training."""
import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import image_pixel as P
import image_kkt as K
import synthetic_confirmation as S


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def final_affine(model):
    layer=model.base.output if isinstance(model.base.output,nn.Linear) else model.base.output[-1]
    assert isinstance(layer,nn.Linear) and layer.bias is not None
    name=next(name for name,module in model.named_modules() if module is layer)
    return name,layer


def state_sha(state,exclude=()):
    h=hashlib.sha256()
    for name,value in state.items():
        if name in exclude:continue
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def capture(model,x,u,ids):
    model.eval();name,layer=final_affine(model);features=[];raw=[]
    def hook(module,inputs):
        assert len(inputs)==1 and inputs[0].ndim==2
        features.append(inputs[0].detach().cpu().numpy().copy())
    handle=layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for chunk in ids.split(128):raw.append(model(x[chunk],u[chunk],raw=True).cpu().numpy())
    finally:handle.remove()
    phi=np.concatenate(features);raw=np.concatenate(raw).astype(np.float64)
    assert phi.dtype==np.float32 and len(phi)==len(ids) and phi.shape[1]==layer.in_features
    assert np.isfinite(phi).all() and np.isfinite(raw).all()
    assert np.array_equal(P.predictions(model,x,u,ids,raw=True),raw)
    replay=[]
    with torch.no_grad():
        for first in range(0,len(phi),128):
            xx=torch.tensor(phi[first:first+128],device=x.device)
            replay.append(F.linear(xx,layer.weight,layer.bias).cpu().numpy())
    assert np.array_equal(np.concatenate(replay).astype(np.float64),raw),'Final-affine cached replay differs from pixel forward'
    scores=P.predictions(model,x,u,ids);cap=np.float32(model.cap)
    assert np.array_equal(scores,np.clip(raw.astype(np.float32),-cap,cap).astype(np.float64))
    return phi,raw,scores


def run(job,args):
    out=args.output/job['name'];out.mkdir(exist_ok=False)
    source=args.data_root/'output'/job['feature_phase']/job['architecture']
    receipt=json.loads((source/'complete.json').read_text())
    assert sha(source/'features.npz')==job['features_sha256']==receipt['features_sha256']
    assert sha(source/'centered_teacher.pt')==job['teacher_sha256']==receipt['centered_teacher_sha256']
    assert job['beta']==.3 and job['kappa']==.95 and job['law']=='image_kkt' and job['head']=='box'
    x,u,prepared_p,training,validation,meta,_=P.prepare(job,args)
    with np.load(source/'features.npz',allow_pickle=False) as original:
        assert np.array_equal(training,original['train_indices']) and np.array_equal(validation,original['validation_indices'])
        assert not np.intersect1d(training,original['teacher_training_indices']).size
        assert not np.intersect1d(validation,original['teacher_training_indices']).size
        canonical=K.construct(original['logits'],math.log(meta['ratio'])/2,job['kappa'])['p']
    assert np.array_equal(prepared_p[:,meta['permutation']],canonical)
    state=torch.load(source/'centered_teacher.pt',map_location='cpu',weights_only=True)
    make=lambda:P.Pixel(job,meta['n'],meta['ratio']).cuda().eval()
    model=make();model.base.load_state_dict(state);model.eval()
    assert all(not v.is_floating_point() or v.dtype==torch.float32 for v in model.state_dict().values())
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    name,layer=final_affine(model);excluded=(name+'.weight',name+'.bias')
    initial=state_sha(model.state_dict());body=state_sha(model.state_dict(),excluded)
    ti=torch.tensor(training,device='cuda');vi=torch.tensor(validation,device='cuda')
    train_phi,train_raw,train_scores=capture(model,x,u,ti)
    val_phi,val_raw,val_scores=capture(model,x,u,vi)
    fresh=make();fresh.base.load_state_dict(state)
    assert np.array_equal(P.predictions(fresh,x,u,ti,raw=True),train_raw)
    assert np.array_equal(P.predictions(fresh,x,u,vi,raw=True),val_raw)
    assert state_sha(model.state_dict())==state_sha(fresh.state_dict())==initial
    assert state_sha(model.state_dict(),excluded)==body
    reports={}
    for split,ids,raw,scores in [('training',training,train_raw,train_scores),('validation',validation,val_raw,val_scores)]:
        reports[split]=S.metrics(canonical[ids],scores,S.oracle(canonical[ids],'box',meta['ratio']),
            'box',job['beta'],raw,meta['ratio'])
    np.savez_compressed(out/'features.npz',training_features=train_phi,validation_features=val_phi,
        training_indices=training,validation_indices=validation,p_training=canonical[training],p_validation=canonical[validation],
        teacher_raw_training=train_raw,teacher_raw_validation=val_raw,
        teacher_box_training=train_scores,teacher_box_validation=val_scores,
        teacher_head_weight=layer.weight.detach().cpu().numpy(),teacher_head_bias=layer.bias.detach().cpu().numpy(),
        feature_mean=train_phi.astype(np.float64).mean(0),feature_std=train_phi.astype(np.float64).std(0).clip(.1))
    S.write(out/'complete.json',dict(job=job,meta=meta,reference=reports,
        training_examples=len(training),validation_examples=len(validation),feature_dimension=train_phi.shape[1],
        features_sha256=sha(out/'features.npz'),source_complete_sha256=sha(source/'complete.json'),
        source_feature_sha256=job['features_sha256'],source_teacher_sha256=job['teacher_sha256'],
        final_affine_name=name,head_state_keys=list(excluded),whole_state_sha256=initial,body_state_sha256=body,
        original_index_order_and_batches128=True,cached_posteriors_are_unpermuted=True,
        cached_affine_raw_replay_bitwise=True,fresh_native_model_raw_replay_bitwise=True,
        parameters_and_buffers_unchanged=True,feature_normalizers_fit='studentB only',
        source_sha256=sha(Path(__file__)),torch=torch.__version__,CUDA=torch.version.cuda,
        official_test_decoded=False,trained_seed=False,optimizer_steps=0,
        scope='Fixed final-affine inputs,not learned head outcomes or a universal FP32 class certificate'))
    assert all(r['mean_D']<=1e-6 and r['near_1e3']>=.99 for r in reports.values()),'Teacher reference preflight failed;do not change learning gates'
    print(json.dumps(dict(architecture=job['architecture'],features=train_phi.shape[1],teacher_validation_D=reports['validation']['mean_D'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--cifar',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    jobs=json.loads(args.queue.read_text())['jobs'];assert len(jobs)==3 and len({j['architecture'] for j in jobs})==3
    failed=[]
    for job in jobs:
        try:run(job,args)
        except Exception:
            failed.append(job['name']);out=args.output/job['name'];out.mkdir(exist_ok=True)
            S.write(out/'failure.json',dict(job=job,traceback=traceback.format_exc()))
    S.write(args.output/'queue_complete.json',dict(jobs=len(jobs),failed=failed,all_successful=not failed))
    if failed:raise RuntimeError('Preserved affine input-cache failures:'+str(failed))

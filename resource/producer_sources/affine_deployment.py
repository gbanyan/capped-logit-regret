"""All15 teacher points or all15 primary raw/tanh hard deployments;zero updates."""
import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import affine_features as A
import image_pixel as P
import synthetic_confirmation as S


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(job,args):
    out=args.output/job['name'];out.mkdir(exist_ok=False);start=time.monotonic()
    assert job['head']=='box' and job['epochs']==0 and job['derived_witness'] and not job['trained_seed']
    assert job['law']=='image_kkt' and job['targets']=='soft' and job['freeze_body']
    cache=args.data_root/'output'/job['affine_feature_phase']/job['architecture']
    receipt=json.loads((cache/'complete.json').read_text())
    assert sha(cache/'features.npz')==job['affine_features_sha256']==receipt['features_sha256']
    assert sha(cache/'complete.json')==job['affine_cache_complete_sha256']
    x,u,p,training,validation,meta,clean=P.prepare(job,args)
    with np.load(cache/'features.npz',allow_pickle=False) as packed:a={name:packed[name] for name in packed.files}
    inverse=np.argsort(meta['permutation'])
    for split,ids in [('training',training),('validation',validation)]:
        assert np.array_equal(a[split+'_indices'],ids) and np.array_equal(p[ids],a['p_'+split][:,inverse])
    model=P.Pixel(job,meta['n'],meta['ratio']).cuda().eval();name,layer=A.final_affine(model)
    exclude=(name+'.weight',name+'.bias');parent=None
    if job['deployment_kind']=='teacher_reference':
        assert sha(clean/'centered_teacher.pt')==job['teacher_sha256']==receipt['source_teacher_sha256']
        model.base.load_state_dict(torch.load(clean/'centered_teacher.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():
            layer.weight.copy_(layer.weight[inverse].clone());layer.bias.copy_(layer.bias[inverse].clone())
        torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},out/'model.pt')
    else:
        assert job['deployment_kind']=='head_transfer' and job['deployment_source_head'] in ('raw','tanh')
        parent=args.data_root/'output'/job['deployment_source_phase']/job['deployment_source_job']
        original=json.loads((parent/'complete.json').read_text());jj=original['job']
        assert jj['head']==job['deployment_source_head'] and jj['affine_arm']=='primary' and jj['epochs']==100
        assert original['meta']==meta and jj['architecture']==job['architecture'] and jj['seed']==job['seed']
        assert sha(parent/'complete.json')==job['deployment_source_complete_sha256']
        assert sha(parent/'model.pt')==job['deployment_source_model_sha256']==original['model_sha256']
        assert sha(parent/'training.npz')==job['deployment_source_training_sha256']
        assert sha(parent/'validation.npz')==job['deployment_source_validation_sha256']
        model.load_state_dict(torch.load(parent/'model.pt',map_location='cpu',weights_only=True))
        shutil.copy2(parent/'model.pt',out/'model.pt');assert sha(out/'model.pt')==original['model_sha256']
    assert all(not value.is_floating_point() or value.dtype==torch.float32 for value in model.state_dict().values())
    assert all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    body=A.state_sha(model.state_dict(),exclude);whole=A.state_sha(model.state_dict())
    assert body==receipt['body_state_sha256'];reports={};reference_delta={}
    for split,ids in [('training',training),('validation',validation)]:
        ix=torch.tensor(ids,device='cuda');raw=P.predictions(model,x,u,ix,raw=True);scores=P.predictions(model,x,u,ix)
        assert np.array_equal(scores,np.clip(raw.astype(np.float32),-np.float32(model.cap),np.float32(model.cap)).astype(np.float64))
        if parent is not None:
            with np.load(parent/(split+'.npz'),allow_pickle=False) as previous:
                assert np.array_equal(raw,previous['raw_scores']) and np.array_equal(p[ids],previous['p'])
                assert np.array_equal(ids,previous['indices'])
        else:
            reference=a['teacher_raw_'+split][:,inverse]
            reference_delta[split]=dict(raw_max_difference=float(abs(raw-reference).max()),
                class_permutation_raw_bitwise=bool(np.array_equal(raw,reference)))
        oo=S.oracle(p[ids],'box',meta['ratio']);reports[split]=S.metrics(p[ids],scores,oo,'box',job['beta'],raw,meta['ratio'])
        if parent is None:assert reports[split]['mean_D']<=1e-6 and reports[split]['near_1e3']>=.99
        np.savez_compressed(out/(split+'.npz'),p=p[ids],scores=scores,raw_scores=raw,indices=ids,oracle_scores=oo['z'],oracle_gap=oo['gap'])
        fresh=P.Pixel(job,meta['n'],meta['ratio']).cuda().eval()
        fresh.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
        assert np.array_equal(P.predictions(fresh,x,u,ix),scores) and np.array_equal(P.predictions(fresh,x,u,ix,raw=True),raw)
        assert A.state_sha(fresh.state_dict())==whole
        del fresh
    assert A.state_sha(model.state_dict())==whole
    S.write(out/'input.json',meta)
    S.write(out/'complete.json',dict(job=job,meta=meta,validation=reports['validation'],training=reports['training'],
        history=[dict(epoch=0,validation=reports['validation'],seconds=time.monotonic()-start)],
        model_sha256=sha(out/'model.pt'),initialization_sha256=whole,body_state_sha256=body,
        final_affine_name=name,trainable_state_keys=[],parameter_and_buffer_bytes_unchanged=True,
        teacher_permutation_comparison=reference_delta,original_raw_scores_bitwise=parent is not None,
        validation_and_training_head_matches_FP32_clamp=True,FP32_checkpoint_dtype_verified=True,
        optimizer_steps=0,optimizer_steps_after_transfer=0,replay_exact=True,raw_replay_exact=True,
        trained_seed=False,derived_witness=True,source_head=job.get('deployment_source_head','teacher'),deployed_head='box',
        parameter_count=sum(v.numel() for v in model.parameters()),trainable_parameter_count=0,
        training_arrays_sha256=sha(out/'training.npz'),validation_arrays_sha256=sha(out/'validation.npz'),
        torch=torch.__version__,CUDA=torch.version.cuda,source_sha256=sha(Path(__file__)),official_test_evaluated=False,
        scope='Actual zero-step FP32 witness/deployment,not independently learned success or projected-gradient training'))
    print(json.dumps(dict(job=job['name'],D=reports['validation']['mean_D'],near=reports['validation']['near_1e3'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--cifar',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    jobs=json.loads(args.queue.read_text())['jobs'];assert len(jobs)==15 and len({j['name'] for j in jobs})==15
    failed=[]
    for job in jobs:
        try:run(job,args)
        except Exception:
            failed.append(job['name']);folder=args.output/job['name'];folder.mkdir(exist_ok=True)
            S.write(folder/'failure.json',dict(job=job,traceback=traceback.format_exc()))
    S.write(args.output/'queue_complete.json',dict(jobs=len(jobs),failed=failed,all_successful=not failed))
    if failed:raise RuntimeError('Preserved affine deployment failures:'+str(failed))

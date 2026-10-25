"""All30 learned affine box endpoints;full B,declared last-affine mask,no updates."""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import affine_features as A
import image_pixel as P
import pixel_box_gradients as G
import synthetic_confirmation as S


def normalized_pullback(gw,gb,mean,std):
    return (gw.astype(np.float64)-gb.astype(np.float64)[:,None]*mean[None,:])/std[None,:]


def run(job,args):
    out=args.output/job['name'];out.mkdir(exist_ok=False)
    source=args.data_root/'output'/job['source_phase']/job['source_job'];r=json.loads((source/'complete.json').read_text());j=r['job']
    assert job['source_phase'] in ('pixel_affine_primary_v1','pixel_affine_secondary_v1')
    assert j['head']=='box' and j['targets']=='soft' and j['initialization']=='random_last_affine'
    assert r['final_affine_only_optimized'] and r['body_parameters_and_buffers_unchanged']
    assert G.sha(source/'model.pt')==job['source_model_sha256']==r['model_sha256']
    assert G.sha(source/'complete.json')==job['source_complete_sha256']
    assert G.sha(source/'validation.npz')==job['source_validation_sha256']
    x,u,p,training,validation,meta,_=P.prepare(j,args);assert meta==r['meta']
    model=P.Pixel(j,meta['n'],meta['ratio']).cuda().eval()
    model.load_state_dict(torch.load(source/'model.pt',map_location='cpu',weights_only=True))
    name,layer=A.final_affine(model);keys=[name+'.weight',name+'.bias']
    for parameter in model.parameters():parameter.requires_grad_(False)
    layer.weight.requires_grad_(True);layer.bias.requires_grad_(True)
    assert sorted(k for k,v in model.named_parameters() if v.requires_grad)==sorted(keys)
    body=A.state_sha(model.state_dict(),keys);assert body==r['body_state_sha256']
    before=G.state_sha(model);ti=torch.tensor(training,device='cuda');vi=torch.tensor(validation,device='cuda')
    with np.load(source/'validation.npz',allow_pickle=False) as saved:
        assert np.array_equal(saved['p'],p[validation]) and np.array_equal(saved['indices'],validation)
        assert np.array_equal(P.predictions(model,x,u,vi),saved['scores'])
        assert np.array_equal(P.predictions(model,x,u,vi,raw=True),saved['raw_scores'])
    raw=P.predictions(model,x,u,ti,raw=True);scores=P.predictions(model,x,u,ti)
    metrics=S.metrics(p[training],scores,S.oracle(p[training],'box',meta['ratio']),'box',j['beta'],raw,meta['ratio'])
    assert abs(metrics['mean_D']-r['training']['mean_D'])<1e-10
    soft,raw32,actual,virtual=G.gradients(model,x,u,ti,p[training],raw,scores_reference=scores)
    soft['trainable_parameter_gradient_linf']=soft.pop('full_parameter_gradient_linf')
    cache=args.data_root/'output'/j['affine_feature_phase']/j['architecture']
    assert G.sha(cache/'features.npz')==j['affine_features_sha256']
    with np.load(cache/'features.npz',allow_pickle=False) as a:mean=a['feature_mean'];std=a['feature_std']
    gw=layer.weight.grad.detach().cpu().numpy();gb=layer.bias.grad.detach().cpu().numpy()
    transformed=normalized_pullback(gw,gb,mean,std)
    assert all(v.grad is None for k,v in model.named_parameters() if k not in keys)
    assert G.state_sha(model)==before and A.state_sha(model.state_dict(),keys)==body
    cap32=np.float32(model.cap);upper=raw32>cap32;lower=raw32<-cap32;strict=upper|lower
    corrective=(upper&(virtual>1e-7))|(lower&(virtual<-1e-7))
    q64=np.exp(S.O.logs(scores));true=p[training];conditional=q64*true.sum(1,keepdims=True)-true
    np.savez_compressed(out/'training_gradients.npz',indices=training,raw_scores=raw32,
        actual_soft_raw_gradient=actual,virtual_soft_raw_gradient=virtual,conditional_soft_score_gradient_FP64=conditional,
        cap32=np.asarray(cap32),native_affine_weight_gradient=gw,native_affine_bias_gradient=gb,
        normalized_chart_weight_gradient_FP64=transformed,normalized_chart_bias_gradient_FP64=gb.astype(np.float64))
    result=dict(job=job,source_training_job=j,source_validation=r['validation'],training=metrics,meta=meta,
        training_examples=len(training),validation_examples=len(validation),validation_replay_bitwise=True,
        soft_eval_gradient=soft,hard_eval_gradient=None,trainable_state_keys=keys,
        trainable_parameter_count=layer.weight.numel()+layer.bias.numel(),
        gradient_scope='Only final native affine parameters;all body parameters frozen,eval mode',
        normalized_chart_gradient_scope='Continuous real coordinate pullback at deployed point;ignores fold rounding,not historical FP32 optimizer gradient',
        normalized_chart_gradient_formula='(gW_native-gb_native*mean)/std;gb_normalized=gb_native',
        strictly_clipped_coordinate_fraction=float(strict.mean()),strict_upper_coordinate_fraction=float(upper.mean()),
        strict_lower_coordinate_fraction=float(lower.mean()),all_train_strictly_clipped_coordinates=np.flatnonzero(strict.all(0)).tolist(),
        all_train_strict_upper_coordinates=np.flatnonzero(upper.all(0)).tolist(),all_train_strict_lower_coordinates=np.flatnonzero(lower.all(0)).tolist(),
        corrective_dead_coordinate_fraction=float(corrective.mean()),corrective_dead_examples=int(corrective.any(1).sum()),
        corrective_virtual_gradient_max=float(abs(virtual[corrective]).max()) if corrective.any() else 0.,
        actual_gradient_max_on_corrective_dead=float(abs(actual[corrective]).max()) if corrective.any() else 0.,
        machine_FP32_corrective_profile=G.corrective_profile(raw32,virtual,cap32),conditional_FP64_corrective_profile=G.corrective_profile(raw32,conditional,cap32),
        machine_vs_conditional_virtual_gradient_max_difference=float(abs(virtual.astype(np.float64)-conditional).max()),
        conditional_gradient_scope='FP64 conditional derivative at same deployed FP32 scores,not FP64 network inference',
        original_per_model_gate=r['validation']['mean_D']<=1e-3 and r['validation']['near_1e3']>=.5,
        parameter_and_buffer_bytes_unchanged=True,checkpoint_state_sha256=before,body_state_sha256=body,
        arrays_sha256=G.sha(out/'training_gradients.npz'),source_sha256=G.sha(Path(__file__)),
        gradient_helper_sha256=G.sha(Path(G.__file__)),optimizer_steps=0,official_test_evaluated=False,trained_seed=False,
        torch=torch.__version__,CUDA=torch.version.cuda,
        scope='All learned affine box endpoints including failed gates;final-affine mask,not all-native or historical stochastic gradients')
    S.write(out/'complete.json',result)
    print(json.dumps(dict(job=job['name'],final_affine_gradient_linf=soft['trainable_parameter_gradient_linf'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--cifar',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    jobs=json.loads(args.queue.read_text())['jobs'];assert len(jobs)==30 and len({j['name'] for j in jobs})==30
    failed=[]
    for job in jobs:
        try:run(job,args)
        except Exception:
            failed.append(job['name']);folder=args.output/job['name'];folder.mkdir(exist_ok=True)
            S.write(folder/'failure.json',dict(job=job,traceback=traceback.format_exc()))
    S.write(args.output/'queue_complete.json',dict(jobs=len(jobs),failed=failed,all_successful=not failed))
    if failed:raise RuntimeError('Preserved final-affine gradient failures:'+str(failed))

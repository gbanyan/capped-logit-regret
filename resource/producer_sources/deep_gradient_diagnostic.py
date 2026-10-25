"""Read-only whole-training gradient and observable-cluster audit of frozen MLPs."""
import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import synthetic_confirmation as S


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(job,args):
    out=args.output/job['name'];out.mkdir(exist_ok=False)
    source=args.data_root/'output'/job['source_phase']/job['source_job']
    record=json.loads((source/'complete.json').read_text());j=record['job']
    assert j['objective']=='CE' and j.get('dtype','float32')=='float32'
    assert sha(source/'model.pt')==record['model_sha256']==job['source_model_sha256']
    assert sha(source/'complete.json')==job['source_complete_sha256']
    options=dict(mode=j['mode'],noise=j.get('noise',.05),shape=j.get('shape',1.),ratio_override=j.get('ratio'))
    x,p,meta=S.data(j['n'],j['beta'],j['seed']*100,j['training'],**options)
    xv,pv,_=S.data(j['n'],j['beta'],j['seed']*100+1,j['validation'],**options)
    assert hashlib.sha256(x.tobytes()+p.tobytes()+xv.tobytes()+pv.tobytes()).hexdigest()==record['data_sha256']
    model=S.Model(j['n'],j['width'],j['depth'],j['head'],meta['ratio']).eval()
    model.load_state_dict(torch.load(source/'model.pt',map_location='cpu',weights_only=True))
    xt=torch.tensor(x);pt=torch.tensor(p,dtype=torch.float32);raws=[];scores=[];actual=[];virtual=[]
    model.zero_grad(set_to_none=True)
    for first in range(0,len(x),512):
        xx=xt[first:first+512];pp=pt[first:first+512];r=model(xx,raw=True);r.retain_grad()
        z=r.clamp(-model.cap,model.cap) if j['head']=='box' else model.cap*torch.tanh(r/model.cap)
        loss=-(pp*F.log_softmax(z,dim=1)).sum()/len(x);loss.backward()
        raws.append(r.detach().numpy());scores.append(z.detach().numpy())
        actual.append((r.grad.detach()*len(x)).numpy())
        virtual.append((torch.softmax(z.detach(),dim=1)*pp.sum(1,keepdim=True)-pp).numpy())
    raw=np.concatenate(raws).astype(np.float64);z=np.concatenate(scores).astype(np.float64)
    actual=np.concatenate(actual).astype(np.float64);virtual=np.concatenate(virtual).astype(np.float64)
    oo=S.oracle(p,j['head'],meta['ratio']);training=S.metrics(p,z,oo,j['head'],j['beta'],raw,meta['ratio'])
    cap32=float(np.float32(model.cap));strict=(raw>cap32)|(raw<-cap32)
    wrong=((raw>cap32)&(oo['z']<model.cap-1e-6))|((raw<-cap32)&(oo['z']>-model.cap+1e-6))
    feasible=np.clip(z,-model.cap,model.cap);d=S.O.loss(p,feasible)-oo['CE'];near=d+oo['gap']+1e-12<=1e-3
    clusters=x[:,:j['n']].argmax(1);cells=[]
    for c in range(j['n']):
        ids=clusters==c;assert ids.any()
        cells.append(dict(cluster=c,examples=int(ids.sum()),mean_D=float(d[ids].mean()),near=float(near[ids].mean()),
            wrong_flat_fraction_by_coordinate=wrong[ids].mean(0).tolist(),
            strict_flat_fraction_by_coordinate=strict[ids].mean(0).tolist(),
            all_coordinates_flat_fraction=float(strict[ids].all(1).mean()),
            actual_raw_gradient_max_by_coordinate=abs(actual[ids]).max(0).tolist(),
            virtual_score_gradient_max_by_coordinate=abs(virtual[ids]).max(0).tolist()))
    readout=model.net[-1]
    gradients=dict(full_parameter_Linf=max(float(v.grad.abs().max()) for v in model.parameters() if v.grad is not None),
        readout_row_Linf=readout.weight.grad.abs().max(1).values.tolist(),readout_bias_gradient=readout.bias.grad.tolist(),
        whole_dataset_strict_saturation_fraction=strict.mean(0).tolist(),
        minimum_raw_distance_beyond_clip_by_coordinate=(abs(raw)-cap32).min(0).tolist(),
        whole_dataset_actual_raw_gradient_max=abs(actual).max(0).tolist(),
        whole_dataset_virtual_score_gradient_max=abs(virtual).max(0).tolist())
    vt=torch.tensor(xv);zv=S.evaluate(model,vt).astype(np.float64);rv=S.evaluate(model,vt,raw=True).astype(np.float64)
    ov=S.oracle(pv,j['head'],meta['ratio']);validation=S.metrics(pv,zv,ov,j['head'],j['beta'],rv,meta['ratio'])
    shutil.copy2(source/'model.pt',out/'model.pt');assert sha(out/'model.pt')==job['source_model_sha256']
    fresh=S.Model(j['n'],j['width'],j['depth'],j['head'],meta['ratio']).eval()
    fresh.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
    assert np.array_equal(S.evaluate(fresh,vt).astype(np.float64),zv)
    np.savez_compressed(out/'validation.npz',p=pv,scores=zv,raw_scores=rv,oracle_scores=ov['z'],oracle_gap=ov['gap'])
    np.savez_compressed(out/'training_gradients.npz',raw=raw,scores=z,actual_raw_gradient=actual,
        virtual_score_gradient=virtual,clusters=clusters,p=p,conditional_D=d)
    history=[dict(epoch=h['epoch'],mean_D=h['validation']['mean_D'],
        wrong_side_saturation=h['validation']['wrong_side_saturation'],
        raw_coordinate_saturation=h['validation']['raw_coordinate_saturation']) for h in record['history']]
    result=dict(job=job,meta=meta,validation=validation,training=training,gradients=gradients,clusters=cells,source_history=history,
        model_sha256=sha(out/'model.pt'),replay_exact=True,derived_witness=True,trained_seed=False,
        optimizer_steps=0,torch=torch.__version__,source_sha256=sha(Path(__file__)),
        scope='Final frozen weights,whole observed training dataset,CPU gradients;not a proof of all historical optimizer states')
    S.write(out/'complete.json',result)
    print(json.dumps(dict(job=job['name'],training_D=training['mean_D'],gradient_Linf=gradients['full_parameter_Linf'],
        bad_clusters=[dict(cluster=c['cluster'],D=c['mean_D'],near=c['near']) for c in cells if c['mean_D']>1e-3])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cifar',type=Path);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for job in json.loads(args.queue.read_text())['jobs']:
        try:run(job,args)
        except Exception:
            out=args.output/job['name'];out.mkdir(exist_ok=True)
            S.write(out/'failure.json',dict(job=job,traceback=traceback.format_exc()));raise

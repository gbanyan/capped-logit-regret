"""Independent full-pixel students and explicitly labeled native warm controls."""
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
import baseline as B
import synthetic_confirmation as S
import image_student as I
import image_kkt as K
from image_data import training_images


class Pixel(nn.Module):
    def __init__(self,job,n,ratio):
        super().__init__();self.base=B.model_for(job['architecture']);self.law=job['law'];self.head=job['head']
        self.cap=math.log(ratio)/2;self.radius=math.sqrt(2)*self.cap
        self.readout=S.Model(n,job['width'],job['depth'],'raw',ratio) if self.law=='teacher_mixed' else None

    def forward(self,x,u,raw=False):
        r=self.base(x)
        if self.readout is not None:r=self.readout(torch.cat([r,(2*u-1)[:,None]],dim=1))
        if raw or self.head=='raw':return r
        if self.head=='box':return r.clamp(-self.cap,self.cap)
        if self.head=='tanh':return self.cap*torch.tanh(r/self.cap)
        if self.head=='ball':return r*(self.radius/r.norm(dim=1,keepdim=True).clamp_min(1e-20)).clamp(max=1)
        raise ValueError(self.head)


def embed_readout(model,state,mean,std,source_permutation,target_permutation):
    """Fold fitted feature normalization into the SAME full-pixel architecture."""
    assert model.readout is not None
    model.readout.load_state_dict(state)
    first=model.readout.net[0];last=model.readout.net[-1];n=len(mean)
    mean=torch.as_tensor(mean,dtype=first.weight.dtype,device=first.weight.device)
    std=torch.as_tensor(std,dtype=first.weight.dtype,device=first.weight.device)
    reorder=np.asarray(source_permutation)[np.argsort(target_permutation)]
    with torch.no_grad():
        original=first.weight[:,:n].clone()
        first.weight[:,:n].copy_(original/std[None,:]);first.bias.sub_((original*(mean/std)[None,:]).sum(1))
        last.weight.copy_(last.weight[reorder].clone());last.bias.copy_(last.bias[reorder].clone())


def predictions(model,x,u,ids,raw=False):
    model.eval();out=[]
    with torch.no_grad():
        for chunk in ids.split(128):out.append(model(x[chunk],u[chunk],raw=raw).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def prepare(job,args):
    folder=args.data_root/'output'/job['feature_phase']/job['architecture']
    receipt=json.loads((folder/'complete.json').read_text());path=folder/'features.npz'
    assert I.sha(path)==job['features_sha256']==receipt['features_sha256']
    assert receipt.get('teacher_student_disjoint') and receipt['CDF_calibration']=='teacher-only halfA'
    a=np.load(path,allow_pickle=False);n=a['logits'].shape[1];beta=job['beta'];ratio=(n-2)*(1-beta)/(2*beta)
    if job['law']=='teacher_mixed':
        _,p,training,validation,meta=I.load(job,args.data_root)
    else:
        assert job['law']=='image_kkt'
        p=K.construct(a['logits'],math.log(ratio)/2,job['kappa'])['p']
        perm=np.random.default_rng(np.random.SeedSequence([job['seed'],19031])).permutation(n)
        pp=np.empty_like(p);pp[:,perm]=p;p=pp
        training=a['train_indices'];validation=a['validation_indices']
        meta=dict(n=n,beta=beta,ratio=ratio,permutation=perm.tolist(),feature_phase=job['feature_phase'],
            features_sha256=job['features_sha256'],law=job['law'])
    images,labels=training_images(job['architecture'],args.data_root,args.cifar)
    assert len(images)==len(p) and not np.intersect1d(training,validation).size
    assert not np.intersect1d(training,a['teacher_training_indices']).size
    x=torch.from_numpy(images).cuda().float().div_(255)
    identity=receipt['identity']
    mean=torch.tensor(identity['mean'],device='cuda').view(1,-1,1,1)
    std=torch.tensor(identity['std'],device='cuda').view(1,-1,1,1);x.sub_(mean).div_(std)
    u=torch.tensor(a['u'],dtype=torch.float32,device='cuda')
    meta.update(law=job['law'],source_identity=identity,teacher_pins=receipt['teacher_pins'],
        training_indices_sha256=hashlib.sha256(training.tobytes()).hexdigest(),
        validation_indices_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),official_test_evaluated=False)
    return x,u,p,training,validation,meta,folder


def train(job,out,args):
    out.mkdir(exist_ok=False);start=time.monotonic()
    x,u,p,training,validation,meta,feature_folder=prepare(job,args)
    n,ratio,beta=meta['n'],meta['ratio'],job['beta'];head=job['head']
    oracle_train=S.oracle(p[training],head,ratio);oracle_val=S.oracle(p[validation],head,ratio)
    torch.manual_seed(job['seed']);model=Pixel(job,n,ratio).cuda()
    if job.get('initialization')=='teacher':
        assert job['law']=='image_kkt'
        receipt=json.loads((feature_folder/'complete.json').read_text())
        assert I.sha(feature_folder/'centered_teacher.pt')==receipt['centered_teacher_sha256']
        model.base.load_state_dict(torch.load(feature_folder/'centered_teacher.pt',map_location='cpu',weights_only=True))
        perm=np.asarray(meta['permutation']);inverse=np.argsort(perm)
        final=model.base.output if isinstance(model.base.output,nn.Linear) else model.base.output[-1]
        with torch.no_grad():
            final.weight.copy_(final.weight[inverse].clone());final.bias.copy_(final.bias[inverse].clone())
    if job.get('initialization')=='feature_witness':
        assert job['law']=='teacher_mixed' and job['epochs']==0
        source=args.data_root/'output'/job['feature_model_phase']/job['feature_model_job']
        witness=json.loads((source/'complete.json').read_text())
        assert I.sha(source/'model.pt')==job['feature_model_sha256']==witness['model_sha256']
        assert witness['meta']['features_sha256']==meta['features_sha256']
        model.base.load_state_dict(torch.load(feature_folder/'centered_teacher.pt',map_location='cpu',weights_only=True))
        arrays=np.load(feature_folder/'features.npz',allow_pickle=False)
        state=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
        embed_readout(model,state,arrays['feature_mean'],arrays['feature_std'],
            witness['meta']['permutation'],meta['permutation'])
    initialization=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    pt=torch.tensor(p,dtype=torch.float32,device='cuda')
    train_ids=torch.tensor(training,device='cuda');val_ids=torch.tensor(validation,device='cuda')
    y=None
    if job['targets']=='hard':
        cumulative=p[training].cumsum(1);cumulative[:,-1]=1
        draw=np.random.default_rng(job['seed']+4141).uniform(size=len(training))
        targets=(draw[:,None]>cumulative).sum(1).astype(np.int64)
        assert targets.max()<n
        y=torch.zeros(len(p),dtype=torch.long,device='cuda');y[train_ids]=torch.tensor(targets,device='cuda')
        np.savez_compressed(out/'sampled_labels.npz',indices=training,labels=targets)
    cifar=job['architecture']=='resnet18';warm=job.get('initialization')=='teacher'
    lr=(.001 if warm else .1) if cifar else (.00001 if warm else .001)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=0) if cifar else torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,job['epochs']),eta_min=lr*.001)
    shuffle=torch.Generator(device='cuda').manual_seed(job['seed']+11);history=[]
    S.write(out/'input.json',meta)
    for epoch in range(job['epochs']+1):
        if epoch%5==0 or epoch==job['epochs']:
            z=predictions(model,x,u,val_ids);raw=predictions(model,x,u,val_ids,raw=True)
            try:report=S.metrics(p[validation],z,oracle_val,head,beta,raw,ratio)
            except Exception:
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'failure_model.pt')
                np.savez_compressed(out/'failure_validation.npz',p=p[validation],scores=z,raw_scores=raw,indices=validation,
                    oracle_scores=oracle_val['z'],oracle_gap=oracle_val['gap'])
                S.write(out/'failure_context.json',dict(job=job,epoch=epoch,history=history,traceback=traceback.format_exc()));raise
            history.append(dict(epoch=epoch,validation=report,seconds=time.monotonic()-start))
            S.write(out/'progress.json',dict(job=job,latest=history[-1]))
            print(json.dumps(dict(job=job['name'],epoch=epoch,mean_D=report['mean_D'],near=report['near_1e3'],seconds=history[-1]['seconds'])),flush=True)
        if epoch==job['epochs']:break
        model.train()
        if job.get('freeze_statistics'):
            for module in model.modules():
                if isinstance(module,(nn.BatchNorm2d,nn.Dropout)):module.eval()
        order=train_ids[torch.randperm(len(train_ids),generator=shuffle,device='cuda')]
        for ids in order.split(128 if cifar else 256):
            optimizer.zero_grad(set_to_none=True);ztrain=model(x[ids],u[ids])
            objective=F.cross_entropy(ztrain,y[ids]) if y is not None else -(pt[ids]*F.log_softmax(ztrain,dim=1)).sum(1).mean()
            objective.backward();optimizer.step()
        scheduler.step()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
    training_z=predictions(model,x,u,train_ids);training_raw=predictions(model,x,u,train_ids,raw=True)
    np.savez_compressed(out/'validation.npz',p=p[validation],scores=z,raw_scores=raw,indices=validation,
        oracle_scores=oracle_val['z'],oracle_gap=oracle_val['gap'])
    fresh=Pixel(job,n,ratio).cuda();fresh.load_state_dict(torch.load(out/'model.pt',weights_only=True,map_location='cpu'))
    assert np.array_equal(predictions(fresh,x,u,val_ids),z)
    S.write(out/'complete.json',dict(job=job,meta=meta,history=history,validation=report,
        training=S.metrics(p[training],training_z,oracle_train,head,beta,training_raw,ratio),
        initialization_sha256=initialization,model_sha256=I.sha(out/'model.pt'),replay_exact=True,
        parameter_count=sum(v.numel() for v in model.parameters()),seconds=time.monotonic()-start,
        torch=torch.__version__,CUDA=torch.version.cuda,official_test_evaluated=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cifar',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;failed=[]
    for job in json.loads(args.queue.read_text())['jobs']:
        try:train(job,args.output/job['name'],args)
        except Exception:
            failed.append(job['name']);folder=args.output/job['name'];folder.mkdir(exist_ok=True)
            S.write(folder/'failure.json',dict(job=job,traceback=traceback.format_exc()));print('FAILED '+job['name'],flush=True)
    S.write(args.output/'queue_complete.json',dict(failed=failed,all_successful=not failed))
    if failed:raise RuntimeError('Preserved full-pixel failures: '+str(failed))

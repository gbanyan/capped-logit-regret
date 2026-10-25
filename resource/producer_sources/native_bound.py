"""Power-of-two real-arithmetic native CNN bound;separate FP64 replay checks."""
import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
import baseline as B


def upper(x):
    x=float(x);assert math.isfinite(x) and x>=0
    return math.frexp(x)[1] if x else None


def lower(x):
    x=float(x);assert x>0 and math.isfinite(x)
    return math.frexp(x)[1]-1


def add(exponents):
    values=[e for e in exponents if e is not None]
    return max(values)+(len(values)-1).bit_length() if values else None


def coefficient(tensor):return upper(tensor.detach().abs().max().item()) if tensor is not None else None


def layer_bound(module,e,path,ledger):
    before=e;kind=type(module).__name__
    if isinstance(module,nn.Sequential):
        for name,child in module.named_children():e=layer_bound(child,e,path+'.'+name,ledger)
        return e
    if isinstance(module,(B.ResNet18,B.FashionCNN,B.FashionMLP)):
        e=layer_bound(module.features,e,path+'.features',ledger)
        return layer_bound(module.output,e,path+'.output',ledger)
    if isinstance(module,B.Block):
        a=layer_bound(module.path,e,path+'.path',ledger);b=layer_bound(module.skip,e,path+'.skip',ledger)
        e=add([a,b])
    elif isinstance(module,(nn.Conv2d,nn.Linear)):
        ew=coefficient(module.weight);eb=coefficient(module.bias)
        fan=module.in_features if isinstance(module,nn.Linear) else (module.in_channels//module.groups)*math.prod(module.kernel_size)
        product=e+ew+(fan-1).bit_length() if e is not None and ew is not None else None
        e=add([product,eb])
    elif isinstance(module,nn.BatchNorm2d):
        assert not module.training and module.running_var is not None and module.running_var.min()>=0
        variance=[lower(module.eps)]
        if module.running_var.min()>0:variance.append(lower(module.running_var.min().item()))
        denominator_exp=max(variance)//2
        eg=coefficient(module.weight) if module.affine else 0
        eb=coefficient(module.bias) if module.affine else None
        em=coefficient(module.running_mean)
        ea=eg-denominator_exp if eg is not None else None
        e=add([ea+e if ea is not None and e is not None else None,eb,
               ea+em if ea is not None and em is not None else None])
    elif isinstance(module,(nn.ReLU,nn.MaxPool2d,nn.AvgPool2d,nn.AdaptiveAvgPool2d,nn.Flatten,nn.Identity)):
        pass
    elif isinstance(module,nn.Dropout):assert not module.training
    else:raise TypeError('Unsupported bound layer '+path+':'+kind)
    ledger.append(dict(path=path,type=kind,input_exponent=before,output_exponent=e))
    return e


def main(raw,out,teacher_phase='baseline_development_v1',centered=False):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);rng=np.random.default_rng(224200);records=[]
    for name in ('resnet18','fashion_cnn','fashion_mlp'):
        folder=raw/'output'/teacher_phase/name
        complete=json.loads((folder/'complete.json').read_text())
        identity=complete['identity'] if centered else json.loads((folder/'input.json').read_text())
        checkpoint=folder/('centered_teacher.pt' if centered else 'model.pt')
        teacher_sha=complete['centered_teacher_sha256'] if centered else complete['model_sha256']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==teacher_sha
        model=B.model_for(name).double().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        input_e=max(add([0,upper(abs(mean))])-lower(std) for mean,std in zip(identity['mean'],identity['std']))
        ledger=[];E=layer_bound(model,input_e,name,ledger);assert E is not None
        n=100 if name=='resnet18' else 10
        caps=[math.log((n-2)*(1-beta)/(2*beta))/2 for beta in (.1,.3,.6)]
        epsilon_e=lower(min(caps))-E-2;assert -1000<epsilon_e<1000
        epsilon=math.ldexp(1.,epsilon_e)
        shape=(3,32,32) if name=='resnet18' else (1,28,28)
        images=np.concatenate([rng.uniform(size=(16,*shape)),np.zeros((1,*shape)),np.ones((1,*shape))])
        x=torch.tensor(images,dtype=torch.float64)
        mean=torch.tensor(identity['mean'],dtype=torch.float64).view(1,-1,1,1)
        std=torch.tensor(identity['std'],dtype=torch.float64).view(1,-1,1,1);x=(x-mean)/std
        with torch.no_grad():original=model(x).numpy()
        final=model.output if isinstance(model.output,nn.Linear) else model.output[-1]
        assert isinstance(final,nn.Linear)
        with torch.no_grad():final.weight.mul_(epsilon);final.bias.mul_(epsilon)
        with torch.no_grad():scaled=model(x).numpy()
        relative=float(abs(scaled/epsilon-original).max()/max(1,float(abs(original).max())))
        assert relative<=1e-12
        gaps=np.partition(original,-2,axis=1);gaps=gaps[:,-1]-gaps[:,-2]
        assert ((original.argmax(1)==scaled.argmax(1))|(gaps<=1e-10)).all()
        torch.save(model.state_dict(),out/(name+'_bayes_scaled.pt'))
        np.savez_compressed(out/(name+'_replay.npz'),raw=original,scaled=scaled,pixels=images)
        records.append(dict(architecture=name,raw_bound_exponent=E,epsilon_exponent=epsilon_e,
            scaled_bound_exponent=E+epsilon_e,min_cap=min(caps),relative_replay_error=relative,
            raw_near_tie_count=int((gaps<=1e-10).sum()),ledger=ledger,
            teacher_sha256=teacher_sha,scaled_checkpoint_sha256=hashlib.sha256((out/(name+'_bayes_scaled.pt')).read_bytes()).hexdigest()))
    result=dict(status='PASS',records=records,torch=torch.__version__,teacher_phase=teacher_phase,centered=centered,
        scope='Exact-real architecture norm argument plus sampled FP64 implementation replay;not global machine-arithmetic proof',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',records=[{k:r[k] for k in ('architecture','raw_bound_exponent','epsilon_exponent','relative_replay_error')} for r in records])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--teacher-phase',default='baseline_development_v1');parser.add_argument('--centered',action='store_true')
    args=parser.parse_args()
    try:main(args.raw,args.output,args.teacher_phase,args.centered)
    except Exception:
        if args.output.is_dir():(args.output/'failure.txt').write_text(traceback.format_exc())
        raise

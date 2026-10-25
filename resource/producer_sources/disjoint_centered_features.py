"""FP64 centered native teacherA/studentB inputs;no official test decode."""
import argparse
import gzip
import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import baseline as B
import oracles as O


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(job,args):
    name=job['architecture'];out=args.output/name;out.mkdir(exist_ok=False)
    teacher=args.data_root/'output/disjoint_teacher_v1'/name
    for file,pin in job['teacher_pins'].items():assert sha(teacher/file)==pin
    identity=json.loads((teacher/'input.json').read_text())
    if name=='resnet18':
        assert hashlib.md5(args.cifar.read_bytes()).hexdigest()=='03b5dce01913d631647c71ecec9e9cb8'
        with tarfile.open(args.cifar) as tar:
            raw=np.frombuffer(tar.extractfile('cifar-100-binary/train.bin').read(),dtype=np.uint8).reshape(-1,3074)
        images=raw[:,2:].copy().reshape(-1,3,32,32);labels=raw[:,1].astype(np.int64);each=50;n=100
    else:
        receipt=json.loads((args.data_root/'input_receipt.json').read_text())
        for record in receipt['fashion']:
            if record['name'].startswith('train'):
                assert hashlib.md5((args.data_root/'fashion'/record['name']).read_bytes()).hexdigest()==record['md5']
        images=np.frombuffer(gzip.decompress((args.data_root/'fashion/train-images-idx3-ubyte.gz').read_bytes()),dtype=np.uint8,offset=16).reshape(-1,1,28,28).copy()
        labels=np.frombuffer(gzip.decompress((args.data_root/'fashion/train-labels-idx1-ubyte.gz').read_bytes()),dtype=np.uint8,offset=8).astype(np.int64);each=600;n=10
    val=np.concatenate([np.flatnonzero(labels==c)[-each:] for c in range(n)]);val.sort()
    pool=np.setdiff1d(np.arange(len(images)),val)
    teacher_train=np.concatenate([pool[labels[pool]==c][:int((labels[pool]==c).sum())//2] for c in range(n)])
    teacher_train.sort();train=np.setdiff1d(pool,teacher_train)
    partition=np.load(teacher/'partition.npz',allow_pickle=False)
    assert np.array_equal(teacher_train,partition['teacher_training'])
    assert np.array_equal(train,partition['student_training']) and np.array_equal(val,partition['validation'])
    assert hashlib.sha256(teacher_train.tobytes()).hexdigest()==identity['train_indices_sha256']
    assert hashlib.sha256(train.tobytes()).hexdigest()==identity['student_indices_sha256']
    assert hashlib.sha256(val.tobytes()).hexdigest()==identity['validation_indices_sha256']
    mean=torch.tensor(identity['mean'],dtype=torch.float64).view(1,-1,1,1)
    std=torch.tensor(identity['std'],dtype=torch.float64).view(1,-1,1,1)
    model=B.model_for(name).double().eval();model.load_state_dict(torch.load(teacher/'model.pt',map_location='cpu',weights_only=True))
    final=model.output if isinstance(model.output,torch.nn.Linear) else model.output[-1]
    assert isinstance(final,torch.nn.Linear)
    with torch.no_grad():
        final.weight.sub_(final.weight.mean(0,keepdim=True));final.bias.sub_(final.bias.mean())
    torch.save(model.state_dict(),out/'centered_teacher.pt')
    logits=[];start=time.monotonic()
    with torch.no_grad():
        for k in range(0,len(images),128):
            x=torch.from_numpy(images[k:k+128]).double().div_(255).sub_(mean).div_(std)
            logits.append(model(x).numpy())
            if k%1280==0:
                B.write(out/'progress.json',dict(latest=dict(epoch=k,
                    validation=dict(extracted=min(k+128,len(images)),total=len(images)))) )
                print(json.dumps(dict(architecture=name,extracted=min(k+128,len(images)),total=len(images))),flush=True)
    logits=np.concatenate(logits);original_reference=np.load(teacher/'validation.npz')['logits'].astype(np.float64)
    reference=original_reference-original_reference.mean(1,keepdims=True)
    difference=float(abs(logits[val]-reference).max());assert difference<=1e-3,difference
    brightness=images.mean(axis=(1,2,3))/255
    quantiles=np.linspace(0,1,1025);knots=np.quantile(brightness[teacher_train],quantiles)
    u=np.interp(brightness,knots,quantiles)
    beta=.3;ratio=(n-2)*(1-beta)/(2*beta);tail=.1*beta;low=tail/(n-2)
    gamma=(1-tail-2*low-.001)*(2*u-1);first=(1-tail+gamma)/2;second=(1-tail-gamma)/2
    w=np.exp(O.logs(logits.astype(np.float64)))
    p=low+(first-low)[:,None]*w+(second-low)[:,None]*np.roll(w,1,axis=1)
    assert p.min()>0 and abs(p.sum(1)-1).max()<1e-12
    oracle=O.box(p,ratio);tied=(p>=ratio*oracle['lower_probability']-1e-12).sum(1)>=2
    np.savez_compressed(out/'features.npz',logits=logits,u=u,brightness=brightness,
        train_indices=train,validation_indices=val,teacher_training_indices=teacher_train,CDF_quantiles=quantiles,CDF_knots=knots,
        feature_mean=logits[train].mean(0),feature_std=logits[train].std(0).clip(.1))
    # Explicit deterministic CPU replay of the entire small heldout validation split.
    replay=[]
    with torch.no_grad():
        for ids in np.array_split(val,max(1,int(np.ceil(len(val)/128)))):
            x=torch.from_numpy(images[ids]).double().div_(255).sub_(mean).div_(std);replay.append(model(x).numpy())
    replay=np.concatenate(replay)
    # Batch shape can change kernel arithmetic;same128-shaped chunks reproduce exactly
    # except the split convention. Report its max difference rather than invent equality.
    replay_difference=float(abs(replay-logits[val]).max());assert replay_difference<1e-4
    result=dict(architecture=name,teacher_pins=job['teacher_pins'],identity=identity,
        student_indices_sha256=identity['student_indices_sha256'],teacher_only_examples=len(teacher_train),
        teacher_student_disjoint=True,teacher_phase='disjoint_teacher_v1',
        teacher_gauge='last affine weights/bias centered in FP64 before inference',
        centered_teacher_sha256=sha(out/'centered_teacher.pt'),
        CDF_fit_indices_sha256=identity['train_indices_sha256'],CDF_calibration='teacher-only halfA',
        CPU_GPU_comparison='CPUFP64 centered network versus row-centered savedGPUFP32 logits',
        CPU_GPU_softmax_max_difference=float(abs(np.exp(O.logs(logits[val]))-np.exp(O.logs(original_reference))).max()),
        examples=len(images),training=len(train),validation=len(val),CPU_GPU_max_difference=difference,
        CPU_GPU_decision_change=float((logits[val].argmax(1)!=reference.argmax(1)).mean()),
        teacher_CPU_validation_accuracy=float((logits[val].argmax(1)==labels[val]).mean()),
        replay_max_difference=replay_difference,CDF_duplicate_knots=int((np.diff(knots)==0).sum()),
        train_oracle_tie_mass=float(tied[train].mean()),validation_oracle_tie_mass=float(tied[val].mean()),
        mean_validation_margin=float((np.partition(p[val],-2,axis=1)[:,-1]-np.partition(p[val],-2,axis=1)[:,-2]).mean()),
        features_sha256=sha(out/'features.npz'),torch=torch.__version__,seconds=time.monotonic()-start,
        official_test_evaluated=False,source_sha256=sha(Path(__file__)))
    B.write(out/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cifar',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for job in json.loads(args.queue.read_text())['jobs']:prepare(job,args)

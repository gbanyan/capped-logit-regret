"""All30 actual raw points against canonical TRAIN-B intervals;15 real label transports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import affine_raw_reference as R
import affine_raw_dual_audit as D


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root,path):return dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path))


def prepare(args):
    root=Path(__file__).resolve().parents[4];base=root/'results/capped_logits_completion_2026-10-22'
    assert not args.output.exists();learners=base/'affine_adequacy_inputs_v1';metadata=json.loads((learners/'summary.json').read_text())
    assert metadata['status']=='ALL_RAW_LEARNER_INPUTS_PREPARED' and len(metadata['jobs'])==30
    reference=base/'cpu_reference_phases/raw_affine_dual_audit_v3';audit=json.loads((reference/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['host']=='gbminipc' and len(audit['verified_jobs'])==3
    assert all(row['same_point_checked'] for row in audit['verified_jobs'])
    artifacts={item['path']:item for item in metadata['source_artifacts']}
    for folder in (learners,reference,base/'affine_raw_training_inputs_v2'):
        for path in folder.rglob('*'):
            if path.is_file():item=artifact(root,path);artifacts[item['path']]=item
    for item in artifacts.values():
        path=root/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    refs=[]
    for architecture in ('resnet18','fashion_cnn','fashion_mlp'):
        folder=reference/'output'/architecture;r=json.loads((folder/'complete.json').read_text())
        refs.append(dict(architecture=architecture,complete_sha256=sha(folder/'complete.json'),arrays_sha256=r['arrays_sha256']))
    context=dict(jobs=metadata['jobs'],references=refs,extractor_sha256=metadata['extractor_sha256'],
        reference_library_sha256=sha(Path(R.__file__)),certificate_library_sha256=sha(Path(D.__file__)),
        artifacts=list(artifacts.values()),cpu_archive_audit=artifact(root,reference/'audit.json'),official_test_decoded=False)
    R.write(args.output,context);print(json.dumps(dict(context_models=30,canonical_problems=3,artifacts=len(artifacts))))


def loss64(T,z):
    maximum=z.max(1,keepdims=True);lse=np.log(np.exp(z-maximum).sum(1))+maximum[:,0]
    return float((T.sum(1)*lse-(T*z).sum(1)).mean())


def analyze(args):
    assert np.finfo(R.L).nmant>52 and not (args.output/'summary.json').exists()
    args.output.mkdir(parents=True,exist_ok=True);context=json.loads(args.context.read_text())
    assert context['reference_library_sha256']==sha(Path(R.__file__))
    assert context['certificate_library_sha256']==sha(Path(D.__file__))
    canonical={};references=[]
    for item in context['references']:
        architecture=item['architecture'];folder=args.reference_root/architecture;r=json.loads((folder/'complete.json').read_text())
        assert sha(folder/'complete.json')==item['complete_sha256'] and sha(folder/'reference.npz')==item['arrays_sha256']==r['arrays_sha256']
        assert r['source_sha256']==sha(Path(D.__file__)) and r['reference_source_sha256']==sha(Path(R.__file__))
        inputs=args.input_root/architecture
        assert sha(inputs/'training_inputs.npz')==r['job']['input_sha256'] and sha(inputs/'complete.json')==r['job']['complete_sha256']
        with np.load(inputs/'training_inputs.npz',allow_pickle=False) as a:
            phi=a['features'].astype(np.float64);T=a['targets'];ids=a['indices'];kept=a['kept_feature_columns']
        with np.load(folder/'reference.npz',allow_pickle=False) as a:b=a['coefficients']
        reduced=np.column_stack([phi[:,kept],np.ones(len(phi))]);score=reduced@b;ce=loss64(T,score)
        assert abs(ce-r['primal']['value'])<1e-7
        canonical[architecture]=dict(X=np.column_stack([phi,np.ones(len(phi))]),reduced=reduced,T=T,ids=ids,b=b,record=r,CE64=ce)
        references.append(dict(architecture=architecture,lower_bound=r['lower_bound'],upper_bound=r['upper_bound'],
            interval_width=r['interval_width'],tight_interval=r['tight_interval'],geometry=r['geometry'],
            training_D_floor_lower=r['training_D_floor_lower'],training_D_floor_upper=r['training_D_floor_upper'],
            reference_coefficient_exceeds_FP32_max=r['reference_coefficient_exceeds_FP32_max'],source_complete_sha256=item['complete_sha256']))
    assert len(canonical)==3;rows=[];transport=[]
    for job in context['jobs']:
        folder=args.learner_root/job['name'];m=json.loads((folder/'complete.json').read_text());c=canonical[job['architecture']]
        assert sha(folder/'complete.json')==job['complete_sha256'] and sha(folder/'native_coefficients.npz')==job['coefficients_sha256']==m['coefficients_sha256']
        assert m['extractor_sha256']==context['extractor_sha256'] and m['actual_p_transport_verified']
        assert m['canonical_input_sha256']==c['record']['job']['input_sha256']
        assert m['canonical_complete_sha256']==c['record']['job']['complete_sha256']
        with np.load(folder/'native_coefficients.npz',allow_pickle=False) as a:w=a['weight'];b=a['bias'];perm=a['permutation']
        assert w.dtype==b.dtype==np.float32 and np.isfinite(w).all() and np.isfinite(b).all()
        n=c['T'].shape[1];assert w.shape==(n,c['X'].shape[1]-1) and b.shape==(n,)
        expected=np.random.default_rng(np.random.SeedSequence([job['seed'],19031])).permutation(n)
        assert np.array_equal(perm,expected) and perm.tolist()==m['actual_label_permutation'];inverse=np.argsort(perm)
        target_sha=hashlib.sha256(c['T'][:,inverse].tobytes()).hexdigest();ids_sha=hashlib.sha256(c['ids'].tobytes()).hexdigest()
        assert target_sha==m['actual_training_p_sha256'] and ids_sha==m['actual_training_indices_sha256']
        # Use FULL original Phi;do not round a constant-column bias fold.
        point=np.vstack([w[perm].astype(np.float64).T,b[perm].astype(np.float64)])
        primal=R.primal_interval(c['X'],c['T'],point);reference=c['record']
        assert primal['upper']>=reference['lower_bound']
        gap_upper=R.upper(R.L(primal['upper'])-R.L(reference['lower_bound']))
        gap_lower=max(0.,R.lower(R.L(primal['lower'])-R.L(reference['upper_bound'])))
        adequate=bool(reference['tight_interval'] and gap_upper<=1e-4)
        rows.append(dict(architecture=job['architecture'],seed=job['seed'],arm=job['arm'],source_phase=m['source_phase'],source_job=m['source_job'],
            source_model_sha256=m['source_model_sha256'],source_complete_sha256=m['source_complete_sha256'],
            real_affine_point_primal=primal,empirical_class_gap_lower=gap_lower,empirical_class_gap_upper=gap_upper,
            raw_adequacy_confirmed=adequate,reference_tight=reference['tight_interval'],
            saved_CUDA_machine_CE=m['actual_training_machine_CE'],saved_CUDA_machine_mean_D=m['actual_training_machine_mean_D'],
            machine_minus_real_affine_CE=m['actual_training_machine_CE']-primal['value'],
            scope='TRAIN-B unbounded real fixed-feature affine class;not wholeNN/population/F32-machine optimizer optimality'))
        if job['arm']=='primary':
            transported=c['b'][:,inverse];assert np.array_equal(transported[:,perm],c['b'])
            ce=loss64(c['T'][:,inverse],c['reduced']@transported);delta=abs(ce-c['CE64']);assert delta<1e-7
            transport.append(dict(architecture=job['architecture'],seed=job['seed'],passed=True,
                actual_target_sha256=target_sha,indices_sha256=ids_sha,coefficient_transport_back_permutation_exact=True,
                transported_reference_CE_difference=delta,source_complete_sha256=m['source_complete_sha256']))
        print(json.dumps(dict(architecture=job['architecture'],arm=job['arm'],seed=job['seed'],gap_upper=gap_upper,adequate=adequate)),flush=True)
    assert len(rows)==30 and len(transport)==15;qualification=[]
    for architecture in sorted(canonical):
        arms={arm:[r for r in rows if r['architecture']==architecture and r['arm']==arm] for arm in ('primary','secondary')}
        assert all(len(value)==5 for value in arms.values())
        qualified={arm:all(r['raw_adequacy_confirmed'] for r in value) for arm,value in arms.items()}
        selected='primary' if qualified['primary'] else 'secondary' if qualified['secondary'] else None
        qualification.append(dict(architecture=architecture,qualified=qualified,Q3_comparison_arm=selected,
            Q1_Q2_primary_arm_unchanged=True,rule='All5 raw seeds;secondary is whole-arm Q3 sensitivity only,never replaces primary learning gates'))
    result=dict(status='COMPLETE_ANALYSIS',objective_scope='trainingB_unbounded_fixed_affine',canonical_problems=references,
        transport_checks=transport,raw_learners=rows,arm_qualification=qualification,artifacts=context['artifacts'],
        cpu_archive_audit=context['cpu_archive_audit'],source_sha256=sha(Path(R.__file__)),
        certificate_source_sha256=sha(Path(D.__file__)),auditor_sha256=sha(Path(__file__)),
        preparation_sha256=context['extractor_sha256'],official_test_decoded=False,longdouble_nmant=np.finfo(R.L).nmant,
        limitations=['Intervals are conditional on the recorded IEEE/libm error model,not a formal machine proof.',
            'Reference coefficients and tiny stored FP32 features define real affine points;possible GPU FTZ and F32 representability are separate.',
            'Saved CUDA loss minus real-affine loss is descriptive;the gap certificate concerns the real point on trainingB only.',
            'No validation/test labels or outcomes select an optimization arm. All30 raw points and15 target transports are retained.'])
    R.write(args.output/'summary.json',result);print(json.dumps(dict(status='COMPLETE_ANALYSIS',raw_points=30,transports=15)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','analyze'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context',type=Path);p.add_argument('--input-root',type=Path);p.add_argument('--reference-root',type=Path);p.add_argument('--learner-root',type=Path)
    args=p.parse_args();prepare(args) if args.mode=='prepare' else analyze(args)

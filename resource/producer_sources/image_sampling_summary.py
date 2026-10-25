"""All final models:fixed-sample paired diagnostics,counts and pre/post gates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import oracles as O
import check_test_release as R
import completion_scope as SC

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parents[1]
MIN_NEAR=30


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values,minimum=2):
    n=len(values);total=float(values.sum())
    if n<minimum:return dict(n=n,sum=total,mean=None,sample_SD=None,standard_error_if_iid=None,suppressed=True)
    sd=float(values.std(ddof=1)) if n>1 else None
    return dict(n=n,sum=total,mean=float(values.mean()),sample_SD=sd,
        standard_error_if_iid=sd/np.sqrt(n) if sd is not None else None,suppressed=False)


def margins(values):
    return dict(n=len(values),quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist() if len(values) else None)


def known(arrays,record,expected):
    p=arrays['p'];head=record['job']['head'];ratio=record['meta']['ratio'];cap=np.log(ratio)/2
    z=arrays['scores'] if head=='raw' else O.project_ball(arrays['scores'],np.sqrt(2)*cap) if head=='ball' else np.clip(arrays['scores'],-cap,cap)
    d=O.loss(p,z)-O.loss(p,arrays['oracle_scores'])
    near=np.maximum(0,d+arrays['oracle_gap']+1e-12)<=1e-3;idx=np.arange(len(p))
    actual=z.argmax(1);raw=arrays['raw_scores'].argmax(1)
    regret=p.max(1)-p[idx,actual];raw_regret=p.max(1)-p[idx,raw]
    increment=regret-raw_regret;disagreement=(actual!=raw)&near
    n=int(near.sum());created=int(((raw_regret==0)&(regret>0)&near).sum());repaired=int(((raw_regret>0)&(regret==0)&near).sum())
    assert abs(float(d.mean())-expected['mean_D'])<1e-10
    assert abs(float(regret.mean())-expected['regret'])<1e-10
    assert float(near.mean())==expected['near_1e3']
    top=np.partition(p,-2,axis=1);bayes_margin=top[:,-1]-top[:,-2]
    return dict(mean_D=describe(d),near_coverage=float(near.mean()),near_examples=n,
        original_gate=bool(d.mean()<=1e-3 and near.mean()>=.5),regret=describe(regret),
        Q2=dict(near_examples=n,degenerate_near=n<MIN_NEAR,
            signed_head_minus_raw=describe(increment[near],MIN_NEAR),
            raw_order_regret=describe(raw_regret[near],MIN_NEAR),
            head_created_Bayes_disagreement_count=created,head_repaired_Bayes_disagreement_count=repaired,
            head_created_fraction=created/n if n>=MIN_NEAR else None,
            head_repaired_fraction=repaired/n if n>=MIN_NEAR else None,
            decision_disagreement_count=int(disagreement.sum()),
            positive_increment_count=int((increment[near]>0).sum()),
            negative_increment_count=int((increment[near]<0).sum()),
            zero_increment_count=int((increment[near]==0).sum()),
            posterior_Bayes_margin_at_disagreements=margins(bayes_margin[disagreement]),
            chosen_class_posterior_difference_at_disagreements=margins(abs(p[idx,raw]-p[idx,actual])[disagreement])))


def main(args):
    assert not args.output.exists();official={}
    if args.official_phase:
        folder=R.DEST/args.official_phase;audit=json.loads((folder/'audit.json').read_text())
        assert audit['status']=='PASS' and audit['expected_jobs']==SC.FINAL_MODELS
        for p in folder.glob('*/complete.json'):
            r=json.loads(p.read_text());j=r['job'];official[j['model_phase'],j['model_job']]=(p,r)
        assert len(official)==SC.FINAL_MODELS
    rows=[];archives=[]
    for phase,kind in R.EVALUATE.items():
        folder=R.DEST/phase;audit=json.loads((folder/'audit.json').read_text())
        assert audit.get('status','PASS')=='PASS' and len(audit['verified_jobs'])==R.REQUIRED[phase]
        manifest=json.loads((folder/'manifest.json').read_text());paths=sorted(folder.glob('*/complete.json'))
        assert len(paths)==R.REQUIRED[phase];archives.append(dict(phase=phase,manifest_sha256=sha(folder/'manifest.json')))
        for p in paths:
            r=json.loads(p.read_text());val=p.parent/'validation.npz'
            assert sha(val)==manifest['files'][str(val.relative_to(folder))]['sha256']
            row=dict(phase=phase,job=p.parent.name,kind=kind,model_sha256=r['model_sha256'],
                complete_sha256=sha(p),validation_sha256=sha(val),
                architecture=r.get('architecture',r.get('job',{}).get('architecture')),
                derived=phase in ('pixel_capacity_replay_v1','native_hard_head_transfer_v1',*SC.AFFINE_DERIVED))
            if kind=='natural':
                row.update(seed=r['seed'],head=r['head'])
                with np.load(val,allow_pickle=False) as a:
                    ce=-O.logs(a['logits'])[np.arange(len(a['labels'])),a['labels']]
                    error=(a['logits'].argmax(1)!=a['labels']).astype(float)
                    row['validation']=dict(CE=describe(ce),error=describe(error))
            else:
                row.update(seed=r['job']['seed'],head=r['job']['head'],training=r['training'],configuration=r['job'])
                with np.load(val,allow_pickle=False) as a:row['validation']=known(a,r,r['validation'])
            if args.official_phase:
                op,o=official[phase,p.parent.name];arrays=op.parent/'test.npz'
                assert o['source_complete_sha256']==sha(p) and o['job']['model_sha256']==r['model_sha256']
                assert sha(arrays)==o['arrays_sha256'];row.update(official_complete_sha256=sha(op),official_arrays_sha256=sha(arrays))
                with np.load(arrays,allow_pickle=False) as a:
                    if kind=='natural':
                        ce=-O.logs(a['logits'])[np.arange(len(a['labels'])),a['labels']]
                        error=(a['logits'].argmax(1)!=a['labels']).astype(float)
                        row['official']=dict(CE=describe(ce),error=describe(error))
                    else:
                        row['official']=known(a,r,o['metrics'])
                        row['validation_to_official_gate_transition']=[row['validation']['original_gate'],row['official']['original_gate']]
            rows.append(row)
    assert len(rows)==SC.FINAL_MODELS and sum(r['kind']!='natural' for r in rows)==588
    raw=ROOT/'results/capped_logits_completion_2026-10-22/cpu_reference_phases/raw_affine_adequacy_v1/output/summary.json'
    q3=json.loads(raw.read_text());assert not q3['official_test_decoded']
    result=dict(status='COMPLETE_ANALYSIS',models=len(rows),known_p_models=588,official_phase=args.official_phase,
        rows=rows,archives=archives,Q3_labels_from_TRAIN_B=q3['arm_qualification'],raw_audit_sha256=sha(raw),
        source_sha256=sha(Path(__file__)),statistics_specification_sha256=sha(BASE/'official_statistics_specification.md'),
        near_minimum_for_additional_conditional_rates=MIN_NEAR,
        limitations=['SD/sqrtN is only a descriptive if-iid diagnostic;fixed class-balanced splits and shared teachers do not establish iid population inference.',
            'Five-seed t summaries describe conditional initialization/data-order/label-permutation variability;no independent dataset replication is implied.',
            'No population sign certificate,p-values or recipe selection;Q2 realized mechanism regret is distinct from SGD worst-minimizer selection.',
            'All original exact endpoint summaries remain archived;N<30 suppression here never modifies original gates or deletes outcomes.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE_ANALYSIS',models=len(rows),official_phase=args.official_phase)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--official-phase');p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

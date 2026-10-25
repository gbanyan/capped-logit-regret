"""All30 raw learners:extract actual FP32 native coefficients and TRAIN-B transport pins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import affine_features as A
import oracles as O

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'results/capped_logits_completion_2026-10-22/phases'
CANON=ROOT/'results/capped_logits_completion_2026-10-22/affine_raw_training_inputs_v2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path):return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path))


def main(output):
    output=output.resolve();assert output.is_relative_to(ROOT) and not output.exists();output.mkdir(parents=True)
    jobs=[];artifacts={}
    for phase,arm in [('pixel_affine_primary_v1','primary'),('pixel_affine_secondary_v1','secondary')]:
        folder=DEST/phase;audit=json.loads((folder/'audit.json').read_text())
        assert audit['status']=='PASS' and len(audit['verified_jobs'])==60
        paths=sorted(folder.glob('*/complete.json'));assert len(paths)==60;count=0
        for path in paths:
            r=json.loads(path.read_text());j=r['job']
            if j['head']!='raw':continue
            count+=1;architecture=j['architecture'];seed=j['seed'];source=path.parent
            assert j['affine_arm']==arm and j['initialization']=='random_last_affine' and j['epochs']==100
            assert r['trained_seed'] and not r['derived_witness'] and r['final_affine_only_optimized']
            assert r['body_parameters_and_buffers_unchanged'] and r['body_eval_mode_throughout']
            assert sha(source/'model.pt')==r['model_sha256'] and sha(source/'training.npz')==r['training_arrays_sha256']
            state=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
            assert all(not v.is_floating_point() or v.dtype==torch.float32 for v in state.values())
            assert all(bool(torch.isfinite(v).all()) for v in state.values())
            keys=[r['final_affine_name']+'.weight',r['final_affine_name']+'.bias']
            assert r['trainable_state_keys']==keys and A.state_sha(state,keys)==r['body_state_sha256']
            cache=DEST/j['affine_feature_phase']/architecture;cache_r=json.loads((cache/'complete.json').read_text())
            assert sha(cache/'complete.json')==j['affine_cache_complete_sha256']
            assert cache_r['body_state_sha256']==r['body_state_sha256']
            canonical=CANON/architecture;canonical_r=json.loads((canonical/'complete.json').read_text())
            assert sha(canonical/'training_inputs.npz')==canonical_r['input_sha256']
            assert canonical_r['source_cache_complete_sha256']==j['affine_cache_complete_sha256']
            perm=np.asarray(r['meta']['permutation']);n=r['meta']['n'];inverse=np.argsort(perm)
            expected=np.random.default_rng(np.random.SeedSequence([seed,19031])).permutation(n)
            assert np.array_equal(perm,expected)
            with np.load(canonical/'training_inputs.npz',allow_pickle=False) as c,np.load(source/'training.npz',allow_pickle=False) as a:
                assert np.array_equal(a['indices'],c['indices'])
                assert np.array_equal(a['p'],c['targets'][:,inverse])
                assert np.isfinite(a['raw_scores']).all() and np.isfinite(a['p']).all()
                actual_p_sha=hashlib.sha256(a['p'].tobytes()).hexdigest();indices_sha=hashlib.sha256(a['indices'].tobytes()).hexdigest()
                actual_machine_CE=float(O.loss(a['p'],a['raw_scores']).mean())
                assert len(a['indices'])==canonical_r['training_examples']
            weight=state[keys[0]].numpy().copy();bias=state[keys[1]].numpy().copy()
            assert weight.shape==(n,canonical_r['original_feature_dimension']) and bias.shape==(n,)
            name=phase+'__'+source.name;out=output/name;out.mkdir()
            np.savez_compressed(out/'native_coefficients.npz',weight=weight,bias=bias,permutation=perm)
            record=dict(name=name,architecture=architecture,arm=arm,seed=seed,source_phase=phase,source_job=source.name,
                source_complete_sha256=sha(path),source_model_sha256=r['model_sha256'],
                source_training_arrays_sha256=r['training_arrays_sha256'],body_state_sha256=r['body_state_sha256'],
                canonical_input_sha256=canonical_r['input_sha256'],canonical_complete_sha256=sha(canonical/'complete.json'),
                actual_training_p_sha256=actual_p_sha,actual_training_indices_sha256=indices_sha,
                actual_label_permutation=r['meta']['permutation'],actual_training_machine_CE=actual_machine_CE,
                actual_training_machine_mean_D=r['training']['mean_D'],
                coefficients_sha256=sha(out/'native_coefficients.npz'),extractor_sha256=sha(Path(__file__)),
                actual_p_transport_verified=True,official_test_decoded=False,
                scope='Actual deployed FP32 native coefficient point on original fixed features;machine-forward CE is descriptive and distinct from exact real-affine CE')
            (out/'complete.json').write_text(json.dumps(record,indent=2)+'\n')
            jobs.append(dict(name=name,architecture=architecture,arm=arm,seed=seed,
                coefficients_sha256=record['coefficients_sha256'],complete_sha256=sha(out/'complete.json')))
            for item in (source/'model.pt',source/'complete.json',source/'training.npz',cache/'complete.json',canonical/'complete.json'):
                value=artifact(item);artifacts[value['path']]=value
        assert count==15
    assert len(jobs)==30 and {(j['architecture'],j['seed'],j['arm']) for j in jobs}=={
        (a,s,arm) for a in ('resnet18','fashion_cnn','fashion_mlp') for s in range(227001,227006) for arm in ('primary','secondary')}
    result=dict(status='ALL_RAW_LEARNER_INPUTS_PREPARED',jobs=jobs,source_artifacts=list(artifacts.values()),
        extractor_sha256=sha(Path(__file__)),numpy=np.__version__,torch=torch.__version__,official_test_decoded=False,
        scope='All30 actual raw native endpoints;no selection by gate,optimization quality or arm')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],models=30)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)

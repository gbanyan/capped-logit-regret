"""Create three training-only canonical problems;preserve exact stored targets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'results/capped_logits_completion_2026-10-22/phases'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    output=output.resolve();assert output.is_relative_to(ROOT)
    assert not output.exists();output.mkdir(parents=True)
    phase=DEST/'affine_feature_cache_v1';audit=json.loads((phase/'audit.json').read_text())
    assert audit['status']=='PASS' and len(audit['checked_inputs'])==3;jobs=[]
    for architecture in ('resnet18','fashion_cnn','fashion_mlp'):
        cache=phase/architecture;r=json.loads((cache/'complete.json').read_text())
        assert sha(cache/'features.npz')==r['features_sha256']
        # Do not decode the archive's validation features or posterior fields.
        with np.load(cache/'features.npz',allow_pickle=False) as a:
            phi=a['training_features'];target=a['p_training'];indices=a['training_indices']
            mean=a['feature_mean'];std=a['feature_std']
        assert phi.dtype==np.float32 and target.dtype==np.float64
        assert len(phi)==len(target)==len(indices) and target.ndim==phi.ndim==2
        assert np.isfinite(phi).all() and np.isfinite(target).all() and target.min()>0
        assert np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>=.1)
        assert len(indices)==len(np.unique(indices)) and abs(target.sum(1)-1).max()<1e-12
        constant=np.all(phi==phi[0],axis=0);removed=np.flatnonzero(constant);kept=np.flatnonzero(~constant)
        values=phi[0,removed]
        assert np.array_equal(phi[:,removed],np.broadcast_to(values,(len(phi),len(removed))))
        folder=output/architecture;folder.mkdir()
        np.savez_compressed(folder/'training_inputs.npz',features=phi,targets=target,indices=indices,
            kept_feature_columns=kept,constant_feature_columns=removed,constant_feature_values=values,
            feature_mean=mean,feature_std=std)
        receipt=dict(architecture=architecture,training_examples=len(phi),original_feature_dimension=phi.shape[1],
            classes=target.shape[1],target_min=float(target.min()),rowmass_max_deviation_from_one=float(abs(target.sum(1)-1).max()),
            targets_renormalized=False,constant_columns=removed.tolist(),constant_values=values.tolist(),
            kept_columns=kept.tolist(),span_equivalence='On TRAIN-B only,each removed column equals its stored constant times the explicit bias column;absorb its coefficient into bias. No near-null direction is truncated.',
            design_scope='Original FP32 features interpreted as exact real numbers with bias;unbounded raw affine class',
            rank_verified=False,rank_scope='Only exact constant columns removed;remaining rank/roundoff bounds require the reference auditor',
            input_sha256=sha(folder/'training_inputs.npz'),source_cache_sha256=r['features_sha256'],
            source_cache_complete_sha256=sha(cache/'complete.json'),source_cache_manifest_sha256=sha(phase/'manifest.json'),
            source_sha256=sha(Path(__file__)),numpy=np.__version__,official_test_decoded=False,
            validation_fields_decoded=False,solver_run=False)
        (folder/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
        jobs.append(dict(architecture=architecture,input_path=str((folder/'training_inputs.npz').relative_to(ROOT)),
            input_sha256=receipt['input_sha256'],complete_path=str((folder/'complete.json').relative_to(ROOT)),
            complete_sha256=sha(folder/'complete.json')))
    result=dict(status='TRAINING_INPUTS_PREPARED',canonical_problems=jobs,official_test_decoded=False,
        objective_scope='trainingB_unbounded_fixed_affine',source_sha256=sha(Path(__file__)),solver_run=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    print(json.dumps(dict(canonical_problems=3,constant_columns={j['architecture']:json.loads((ROOT/j['complete_path']).read_text())['constant_columns'] for j in jobs})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)

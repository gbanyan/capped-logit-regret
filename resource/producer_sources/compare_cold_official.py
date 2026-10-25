"""Compare actual restored teacher3/model648 outputs with cold-retrieved originals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_arrays(candidate,reference):
    rows=[]
    with np.load(candidate,allow_pickle=False) as a,np.load(reference,allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for name in sorted(a.files):
            aa,bb=a[name],b[name]
            assert aa.dtype==bb.dtype and aa.shape==bb.shape
            assert aa.tobytes()==bb.tobytes(),str(candidate)+':'+name
            rows.append(dict(name=name,shape=list(aa.shape),dtype=str(aa.dtype),
                array_bytes_sha256=hashlib.sha256(aa.tobytes()).hexdigest(),bitwise=True))
    return dict(arrays=rows,all_array_values_bitwise=True,serialized_NPZ_bytes_equal=sha(candidate)==sha(reference))


def main(args):
    assert not (args.output/'summary.json').exists()
    queue=json.loads(args.queue.read_text());assert len(queue['jobs'])==1;j=queue['jobs'][0]
    restored=args.data_root/'restored_origin_v1';provenance=json.loads((restored/'restoration_provenance.json').read_text())
    bridge=json.loads((restored/'bridge_verified.json').read_text())
    assert bridge['status']=='ALL_TRANSFERRED_ORIGIN_BYTES_VERIFIED'
    assert sha(restored/'restoration_provenance.json')==bridge['cold_provenance_sha256']
    assert bridge['commit']==provenance['commit'] and provenance['model_slots']==648
    reference=restored/'output';actual=args.data_root/'output'
    release=reference/'sealed_test_release.json';release_sha=sha(release)
    assert release_sha==provenance['release_sha256']==j['release_sha256']
    refs={(r['model_phase'],r['model_job']):r for r in json.loads(release.read_text())['frozen_model_refs']}
    assert len(refs)==648;features=[];models=[]
    for architecture in ('resnet18','fashion_cnn','fashion_mlp'):
        aa=actual/j['feature_replay_phase']/architecture;bb=reference/j['reference_feature_phase']/architecture
        a=json.loads((aa/'complete.json').read_text());b=json.loads((bb/'complete.json').read_text())
        assert a['job']==b['job'] and a['release_sha256']==b['release_sha256']==release_sha
        assert a['source_sha256']==b['source_sha256'] and a['source_feature_receipt']==b['source_feature_receipt']
        assert a['data_identity']==b['data_identity'] and a['indices_sha256']==b['indices_sha256']
        assert a['replay_exact'] and b['replay_exact']
        assert sha(aa/'features.npz')==a['features_sha256'] and sha(bb/'features.npz')==b['features_sha256']
        result=compare_arrays(aa/'features.npz',bb/'features.npz')
        features.append(dict(architecture=architecture,**result))
    expected=json.loads((reference/j['reference_evaluate_phase']/'source/queue.json').read_text())['jobs']
    assert len(expected)==648 and {(x['model_phase'],x['model_job']) for x in expected}==set(refs)
    folder=actual/j['evaluate_replay_phase'];done=json.loads((folder/'queue_complete.json').read_text())
    assert done['all_successful'] and not done['failed'] and done['jobs']==648
    assert len(list(folder.glob('*/complete.json')))==648 and not list(folder.glob('*/failure.json'))
    for job in expected:
        aa=folder/job['name'];bb=reference/j['reference_evaluate_phase']/job['name']
        a=json.loads((aa/'complete.json').read_text());b=json.loads((bb/'complete.json').read_text())
        assert a['job']==b['job']==job
        assert a['source_model_record']==b['source_model_record']
        assert a['source_complete_sha256']==b['source_complete_sha256']==refs[job['model_phase'],job['model_job']]['complete_sha256']
        assert a['release_sha256']==b['release_sha256']==release_sha
        assert a['source_sha256']==b['source_sha256'] and a['metrics']==b['metrics']
        assert a['data_identity']==b['data_identity'] and a['indices_sha256']==b['indices_sha256']
        assert a['replay_exact'] and b['replay_exact']
        assert sha(aa/'test.npz')==a['arrays_sha256'] and sha(bb/'test.npz')==b['arrays_sha256']
        result=compare_arrays(aa/'test.npz',bb/'test.npz')
        models.append(dict(name=job['name'],model_phase=job['model_phase'],model_job=job['model_job'],
            model_sha256=job['model_sha256'],source_complete_sha256=a['source_complete_sha256'],
            fresh_complete_sha256=sha(aa/'complete.json'),retrieved_complete_sha256=sha(bb/'complete.json'),
            fresh_arrays_sha256=a['arrays_sha256'],retrieved_arrays_sha256=b['arrays_sha256'],metrics_equal=True,**result))
    assert len(models)==648
    launches={name:json.loads((actual/name/'launch.json').read_text())
        for name in (j['feature_replay_phase'],j['evaluate_replay_phase'])}
    for r in launches.values():
        assert r['scientific_input_root'].endswith('/restored_origin_v1')
        assert r['release_sha256']==release_sha
    summary=dict(status='ALL_RESTORED_OFFICIAL_PATHS_BITWISE',teacher_input_packs=3,models=648,
        cold_commit=bridge['commit'],cold_provenance_sha256=bridge['cold_provenance_sha256'],
        features=features,model_rows=models,replay_launches=launches,release_sha256=release_sha,
        all_metrics_equal=True,all_saved_array_values_bitwise=True,optimizer_updates=0,
        source_sha256=sha(Path(__file__)),numpy=np.__version__,
        scope='Actual teacher input and all648 model official inference replay from cold-origin assets;terminal/source/runtime checks must additionally be archived by root;not additional trained seeds or R1-R7 completion')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],teacher_inputs=3,models=648)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cifar',type=Path,required=True);main(p.parse_args())

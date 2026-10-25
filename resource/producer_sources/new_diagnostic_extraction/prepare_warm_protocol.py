"""Extract original warm hard-label protocol and compare saved paired labels."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASE = ROOT / 'results/capped_logits_completion_2026-10-22/phases/pixel_kkt_warm_v1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest = json.loads((PHASE / 'source/source_manifest.json').read_text())
    for name in ('producer.py', 'queue.json', 'protocol.md'):
        assert sha(PHASE / 'source' / name) == manifest['inputs'][name]
    jobs = json.loads((PHASE / 'source/queue.json').read_text())['jobs']
    assert len(jobs) == 30 and all(j['targets'] == 'hard' for j in jobs)
    groups = {}
    rows = []
    for job in jobs:
        root = PHASE / job['name']
        complete = json.loads((root / 'complete.json').read_text())
        assert complete['job'] == job
        label_path = root / 'sampled_labels.npz'
        with np.load(label_path, allow_pickle=False) as a:
            assert set(a.files) == {'indices','labels'}
            indices, labels = a['indices'], a['labels']
            assert indices.shape == labels.shape
            digest = {k: dict(dtype=str(a[k].dtype), shape=list(a[k].shape),
                             C_order_sha256=hashlib.sha256(a[k].tobytes()).hexdigest()) for k in a.files}
        key = (job['architecture'], job['seed'])
        groups.setdefault(key, []).append(digest)
        rows.append(dict(job=job, source_record=str((root/'complete.json').relative_to(ROOT)),
                         source_record_sha256=sha(root/'complete.json'),
                         sampled_labels=str(label_path.relative_to(ROOT)), sampled_labels_sha256=sha(label_path),
                         arrays=digest, examples=len(labels), initialization_sha256=complete['initialization_sha256']))
    assert len(groups) == 15 and all(len(v) == 2 and v[0] == v[1] for v in groups.values())
    out = dict(kind='NEW read-only extraction/paired saved-label comparison; no generation or training',
               models=30, paired_blocks=15, all_pairs_indices_and_labels_byte_identical=True,
               original_source_manifest=dict(path=str((PHASE/'source/source_manifest.json').relative_to(ROOT)),sha256=sha(PHASE/'source/source_manifest.json')),
               source_inspected_protocol=dict(targets='hard', objective='F.cross_entropy with one saved label per B image',
                                              draw_seed='job seed + 4141', draw_frequency='once before epoch loop; reused each epoch',
                                              shuffle_seed='job seed + 11', paired_modes='same architecture/seed, posterior/initialization/labels/shuffle seed; only BN/dropout statistics mode differs'),
               limits=['Source/hash and saved-label checks, not independent validation of original random draws or optimizer execution.',
                       'Sampled-label finite-sample risk differs from posterior CE at a fixed model, despite equality in expectation over labels.',
                       'No source scientific summary or original record was modified.'], rows=rows)
    (HERE/'response_draft/warm_protocol_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(models=len(rows),paired_blocks=len(groups),same_labels_all_pairs=True)))

if __name__ == '__main__':
    main()

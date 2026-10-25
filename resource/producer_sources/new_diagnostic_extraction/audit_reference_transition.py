"""New read-only byte comparison of existing reference arrays; no solver runs."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / 'results/capped_logits_completion_2026-10-22'
CODE = ROOT / 'research/next_exploration/capped_logits_completion_2026-10-22/code'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    rows = []
    for arch in ('resnet18', 'fashion_cnn', 'fashion_mlp'):
        roots = [RESULTS / f'cpu_reference_phases/{v}/output/{arch}' for v in
                 ('raw_affine_reference_v2', 'raw_affine_dual_audit_v3')]
        records = [json.loads((p / 'complete.json').read_text()) for p in roots]
        for p, r in zip(roots, records):
            assert sha(p / 'reference.npz') == r['arrays_sha256']
        before, after = records
        assert sha(roots[0] / 'complete.json') == after['parent_complete_sha256']
        assert before['arrays_sha256'] == after['parent_arrays_sha256']
        assert sha(CODE / 'affine_raw_reference.py') == before['source_sha256'] == after['reference_source_sha256']
        assert sha(CODE / 'affine_raw_dual_audit.py') == after['source_sha256']
        fields = {}
        with np.load(roots[0] / 'reference.npz', allow_pickle=False) as a, np.load(roots[1] / 'reference.npz', allow_pickle=False) as b:
            for key in ('coefficients', 'preconditioner'):
                x, y = a[key], b[key]
                same = x.shape == y.shape and x.dtype == y.dtype and x.tobytes(order='C') == y.tobytes(order='C')
                assert same
                fields[key] = dict(shape=list(x.shape), dtype=str(x.dtype), byte_identical=same,
                                   C_order_bytes_sha256=hashlib.sha256(x.tobytes(order='C')).hexdigest())
        assert after['same_point_no_optimizer_updates'] and after['optimizer_steps'] == 0
        assert before['upper_bound'] == after['upper_bound']
        rows.append(dict(architecture=arch, arrays=fields,
                         original_records=[dict(path=str((p / 'complete.json').relative_to(ROOT)), sha256=sha(p / 'complete.json')) for p in roots],
                         parent_link_hashes_match=True, source_hashes_match=True,
                         upper_bound=after['upper_bound'], v2_width=before['interval_width'],
                         v3_width=after['interval_width'], v2_tight=before['tight_interval'],
                         v3_tight=after['tight_interval'], saved_additional_primal_optimizer_steps=after['optimizer_steps']))
    out = dict(kind='NEW local read-only comparison of existing saved arrays; not historical execution',
               numpy=np.__version__, rows=rows, all_coefficients_and_maps_byte_identical=True,
               criteria=dict(reference_width=1e-6, learner_gap=1e-4),
               criteria_evidence='Original v2/v3 producer width guards and original reference/adequacy protocols; no inferred timestamp ordering.',
               limits=['No optimizer, inference or interval arithmetic rerun.',
                       'Original runtime conditional arithmetic certificate remains conditional on its IEEE/libm model.',
                       'Zero additional primal optimization does not mean no dual-candidate arithmetic.',
                       'Original 30 learner outcomes and earlier wide intervals are retained.'])
    (HERE / 'response_draft/reference_transition_audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({r['architecture']: [r['v2_width'], r['v3_width']] for r in rows}))

if __name__ == '__main__':
    main()

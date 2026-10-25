"""Extract existing origin/runtime/loader receipts; no host access or loading."""
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT/'results/capped_logits_completion_2026-10-22'
CODE = ROOT/'research/next_exploration/capped_logits_completion_2026-10-22/code'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    original = RESULTS/'cold_origin_final_audit_attempt001/summary.json'
    value = json.loads(original.read_text())
    binaries = value['checked_binaries']
    assert len(binaries) == value['binary_paths'] == 4713
    assert len({r['sha256'] for r in binaries}) == value['unique_LFS_objects'] == 4582
    assert sum(r['bytes'] for r in binaries) == value['bytes'] == 24778294586
    assert len(value['manifests']) == 59
    assert sha(CODE/'verify_restored_completion.py') == value['source_sha256']
    receipt_paths = ['cold_origin_initial_audit_attempt001/initial_empty_store_receipt.json',
                     'cold_origin_start_attempt001/summary.json', 'cold_origin_finalize_attempt001/summary.json',
                     'runtime_inventory_attempt001/summary.json', 'restore_native_runtime_attempt001/summary.json',
                     'shutdown_attempt001/receipt.json']
    receipts = {name: json.loads((RESULTS/name).read_text()) for name in receipt_paths}
    initial, start, final = [receipts[n] for n in receipt_paths[:3]]
    assert initial['no_existing_checkout_or_LFS_store']
    assert initial['source_sha256'] == start['source_sha256'] == final['source_sha256'] == sha(CODE/'cold_retrieve_completion.py')
    assert start['root'] == final['root'] == value['root']
    assert start['isolated_store'] == final['isolated_store'] == value['isolated_lfs_storage']
    assert final['commit'] == value['commit'] and final['final_assets_covered']
    fields = ['path','sha256','bytes','worktree_materialized','checkpoint_safely_loaded','tensor_count','all_tensors_finite','arrays_safely_loaded']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, lineterminator='\n'); writer.writeheader()
    for row in binaries:
        writer.writerow({k:row.get(k,'') for k in fields})
    csv_path = HERE/'response_draft/restored_binary_roster.csv'
    csv_path.write_bytes(stream.getvalue().encode())
    runtime = receipts['runtime_inventory_attempt001/summary.json']
    out = dict(kind='NEW extraction of original saved origin/runtime/loading records; no fresh restoration, tensor parsing or inference',
               original_audit=dict(path=str(original.relative_to(ROOT)), sha256=sha(original)),
               original_metadata={k:v for k,v in value.items() if k!='checked_binaries'},
               binary_roster=dict(file=csv_path.name, sha256=sha(csv_path), rows=len(binaries)),
               original_receipts=[dict(path=str((RESULTS/n).relative_to(ROOT)),sha256=sha(RESULTS/n)) for n in receipt_paths],
               all_receipt_links_match=True, GPU_image_id=runtime['image_id'], GPU_versions=runtime['versions'],
               CPU_loading_versions=dict(torch=value['torch'],numpy=value['numpy']),
               checkpoint_safe_load_records=sum(bool(r.get('checkpoint_safely_loaded')) for r in binaries),
               array_safe_load_records=sum(bool(r.get('arrays_safely_loaded')) for r in binaries),
               source_inspected_loading=dict(torch='torch.load(map_location=cpu, weights_only=True)',numpy='numpy.load(allow_pickle=False)'),
               limits=['Saved receipts and source inspection, not independent authentication of historical execution.',
                       'Bytes sum over binary paths, not unique-object storage bytes.',
                       'CPU serialization-audit versions differ from the pinned GPU inference runtime.',
                       'Shutdown receipt records accepted command and SSH unreachability, not direct physical power observation.',
                       'Current live host state is unknown; no host was contacted.'])
    (HERE/'response_draft/restoration_runtime_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['all_receipt_links_match','checkpoint_safe_load_records','array_safe_load_records','GPU_image_id']}))

if __name__ == '__main__':
    main()

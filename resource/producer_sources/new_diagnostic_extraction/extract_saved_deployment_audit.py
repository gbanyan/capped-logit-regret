"""Extract existing fixed-affine deployment records, without model execution."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASES = ROOT / "results/capped_logits_completion_2026-10-22/phases"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(metrics):
    return metrics["mean_D"] <= 1e-3 and metrics["near_1e3"] >= .5


def main():
    records = []
    for phase in ("pixel_affine_primary_v1", "pixel_affine_secondary_v1"):
        paths = sorted((PHASES / phase).glob("*/complete.json"))
        assert len(paths) == 60
        for path in paths:
            value = json.loads(path.read_text())
            job = value["job"]
            receipt_path = PHASES / job["affine_feature_phase"] / job["architecture"] / "complete.json"
            receipt = json.loads(receipt_path.read_text())
            assert sha(receipt_path) == job["affine_cache_complete_sha256"]
            assert receipt["body_state_sha256"] == value["body_state_sha256"]
            record = {k: value[k] for k in (
                "job", "body_state_sha256", "body_parameters_and_buffers_unchanged",
                "body_eval_mode_throughout", "final_affine_name", "trainable_state_keys",
                "final_affine_only_optimized", "folding", "cached_vs_deployed", "source_sha256")}
            record.update(source=str(path.relative_to(ROOT)), source_record_sha256=sha(path),
                          cache_receipt=str(receipt_path.relative_to(ROOT)), cache_receipt_sha256=sha(receipt_path),
                          cache_receipt_body_state_sha256=receipt["body_state_sha256"])
            records.append(record)
    splits = [v for r in records for v in r["cached_vs_deployed"].values()]
    producer = ROOT / "research/next_exploration/capped_logits_completion_2026-10-22/code/affine_student.py"
    output = dict(
        kind="Extraction/recalculation from existing saved records; no neural or optimizer execution",
        source_baseline="d1bef6621301666371c8367430479518f1278925",
        models=len(records), split_records=len(splits),
        all_receipt_and_reported_common_body_hashes_equal=True,
        all_report_unchanged_body=all(r["body_parameters_and_buffers_unchanged"] for r in records),
        all_report_eval_mode=all(r["body_eval_mode_throughout"] for r in records),
        maximum_cached_native_raw_difference=max(v["raw_max_difference"] for v in splits),
        maximum_cached_native_score_difference=max(v["score_max_difference"] for v in splits),
        maximum_decision_disagreement=max(v["decision_disagreement"] for v in splits),
        nonzero_decision_disagreement_splits=sum(v["decision_disagreement"] != 0 for v in splits),
        cached_native_Q1_gate_mismatch_splits=sum(gate(v["cached_metrics"]) != gate(v["deployed_metrics"]) for v in splits),
        producer=dict(path=str(producer.relative_to(ROOT)), sha256=sha(producer),
                      all_records_match_current_frozen_producer=all(r["source_sha256"] == sha(producer) for r in records),
                      inspected_lines="79; 108; 126; 128; 134-137: receipt and post-fold/reload/final body assertions; cached/native saved fields"),
        limitations=[
            "Source inspection and saved-record consistency, not newly recomputed tensor hashes or inference.",
            "Original model records retain one common body hash plus an unchanged flag, not a separate final-hash string.",
            "The source asserts the same hash after folding, reloading and final evaluation; assertions alone are not independent execution proof.",
            "Nonzero cache/native differences are preserved; actual native gates remain the scientific endpoint.",
            "No official cached/native comparison is inferred from training/validation records."
        ], records=records)
    destination = HERE / "response_draft/saved_deployment_audit.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("records", "limitations", "producer")}))


if __name__ == "__main__":
    main()

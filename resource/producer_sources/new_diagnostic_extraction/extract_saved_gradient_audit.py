"""Extract existing checkpoint gradient evidence; no backpropagation or model run."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASES = ROOT / "results/capped_logits_completion_2026-10-22/phases"
CODE = ROOT / "research/next_exploration/capped_logits_completion_2026-10-22/code"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    official_path = PHASES.parent / 'official_summary_attempt001/summary.json'
    official_rows = json.loads(official_path.read_text())['rows']
    official = {(r['model_phase'], r['model_job']): r for r in official_rows}
    assert len(official) == len(official_rows)
    for phase, expected, producer, mask, norm_key in (
        ("pixel_box_gradient_diagnostic_v1", 69, "pixel_box_gradients.py", "full native parameters", "full_parameter_gradient_linf"),
        ("affine_box_gradient_diagnostic_v1", 30, "affine_box_gradients.py", "final native affine parameters only", "trainable_parameter_gradient_linf"),
    ):
        paths = sorted((PHASES / phase).glob("*/complete.json"))
        assert len(paths) == expected
        for path in paths:
            value = json.loads(path.read_text())
            gradient = value["soft_eval_gradient"]
            norm = gradient[norm_key]
            assert math.isfinite(norm)
            job = value['job']
            endpoint = official[(job['source_phase'], job['source_job'])]
            model_path = PHASES / job['source_phase'] / job['source_job'] / 'model.pt'
            assert sha(model_path) == job['source_model_sha256']
            record = {key: value[key] for key in (
                "job", "training_examples", "validation_examples", "validation_replay_bitwise",
                "soft_eval_gradient", "hard_eval_gradient", "checkpoint_state_sha256",
                "parameter_and_buffer_bytes_unchanged", "source_sha256", "optimizer_steps", "scope")}
            record.update(source=str(path.relative_to(ROOT)), source_record_sha256=sha(path),
                          diagnostic_phase=phase, parameter_mask=mask,
                          mask_description_basis="Saved norm-field identity and inspected producer source; not newly enumerated tensors",
                          conditional_eval_gradient_linf=norm, norm_field=norm_key,
                          eval_mode_source="Matching producer explicitly calls model.eval()",
                          producer_path=str((CODE / producer).relative_to(ROOT)),
                          producer_matches_saved_hash=sha(CODE / producer) == value["source_sha256"])
            record.update(checkpoint_file=str(model_path.relative_to(ROOT)),
                          current_checkpoint_file_sha256=job['source_model_sha256'],
                          official_endpoint_job=endpoint['job'],
                          official_mean_D=endpoint['mean_D'],
                          official_near_coverage=endpoint['near_coverage'],
                          official_Q1_pass=endpoint['combined_gate'],
                          official_gate_scope='Separate official-input Q1, not the training-B gradient objective',
                          loss_and_network_precision='Producer source: FP32 network/log_softmax, posterior targets cast to FP32, minibatch gradients accumulated with sum/full-B scaling; separate conditional score derivative is FP64',
                          norm_zero_criterion='Recorded finite L-infinity norm equals exactly zero, without a small-gradient threshold')
            for key in ("gradient_scope", "trainable_state_keys", "trainable_parameter_count", "gradient_helper_sha256", "conditional_gradient_scope"):
                if key in value:
                    record[key] = value[key]
            records.append(record)
    output = dict(
        kind="Extraction of existing 99-checkpoint diagnostics; no gradient/model execution",
        source_baseline="d1bef6621301666371c8367430479518f1278925",
        models=len(records), native_parameter_masks=69, final_affine_parameter_masks=30,
        minimum_recorded_gradient_linf=min(r["conditional_eval_gradient_linf"] for r in records),
        maximum_recorded_gradient_linf=max(r["conditional_eval_gradient_linf"] for r in records),
        exact_zero_recorded_norms=sum(r["conditional_eval_gradient_linf"] == 0 for r in records),
        all_saved_gradient_finite_flags=all(r["soft_eval_gradient"]["all_parameter_gradients_finite"] for r in records),
        all_producers_match_saved_hashes=all(r["producer_matches_saved_hash"] for r in records),
        all_report_zero_optimizer_updates=all(r["optimizer_steps"] == 0 for r in records),
        official_summary=dict(path=str(official_path.relative_to(ROOT)), sha256=sha(official_path)),
        official_Q1_passes=sum(r['official_Q1_pass'] for r in records),
        official_Q1_failures=sum(not r['official_Q1_pass'] for r in records),
        limitations=["Stored conditional full-B evaluation-mode gradient diagnostics, not historical training gradients.",
                     "Full-native and final-affine parameter masks are different estimands.",
                     "Producer-source/matching-record inspection, not independent gradient recomputation.",
                     "Nonzero checkpoint gradients do not identify every training failure mechanism or prove convergence."],
        records=records)
    (HERE / "response_draft/saved_gradient_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("records", "limitations")}))


if __name__ == "__main__":
    main()

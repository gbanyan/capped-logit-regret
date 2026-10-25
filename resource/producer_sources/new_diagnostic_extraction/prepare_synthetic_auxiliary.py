"""Document existing synthetic diagnostics and later-wave outcomes; no new runs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results/capped_logits_completion_2026-10-22"
CODE = ROOT / "research/next_exploration/capped_logits_completion_2026-10-22/code"
OUT = HERE / "response_draft"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    diagnostics = []
    for seed in (223001, 223003):
        path = RESULTS / f"phases/deep_gradient_diagnostic_v1/deep_box_s{seed}/complete.json"
        value = json.loads(path.read_text())
        source = RESULTS / "phases" / value["job"]["source_phase"] / value["job"]["source_job"] / "complete.json"
        original = json.loads(source.read_text())
        assert sha(source) == value["job"]["source_complete_sha256"]
        assert original["model_sha256"] == value["model_sha256"] == value["job"]["source_model_sha256"]
        assert sha(CODE / "deep_gradient_diagnostic.py") == value["source_sha256"]
        g = value["gradients"]
        diagnostics.append(dict(
            seed=seed, source=str(path.relative_to(ROOT)), source_sha256=sha(path),
            original_model_source=str(source.relative_to(ROOT)), original_record_sha256=sha(source),
            checkpoint_model_sha256=value["model_sha256"], input_split="whole observed training set",
            examples=original["job"]["training"], loss="conditional posterior cross-entropy", parameter_mask="full model parameters",
            full_parameter_Linf=g["full_parameter_Linf"],
            minimum_coordinate_strict_saturation_fraction=min(g["whole_dataset_strict_saturation_fraction"]),
            minimum_distance_beyond_clip=min(g["minimum_raw_distance_beyond_clip_by_coordinate"]),
            validation_raw_saturation_fraction=original["validation"]["raw_coordinate_saturation"],
            scope=value["scope"], optimizer_steps=value["optimizer_steps"]))
    wave = []
    for path in sorted((RESULTS / "phases/synthetic_deep_replication_v1").glob("*/complete.json")):
        value = json.loads(path.read_text())
        metrics = value["validation"]
        wave.append(dict(job=value["job"], source=str(path.relative_to(ROOT)), source_sha256=sha(path),
                         data_sha256=value["data_sha256"], initialization_sha256=value["initialization_sha256"],
                         model_sha256=value["model_sha256"], validation=metrics,
                         gate=metrics["mean_D"] <= 1e-3 and metrics["near_1e3"] >= .5))
    assert len(wave) == 10 and {r["job"]["seed"] for r in wave} == set(range(223501, 223506))
    for seed in range(223501, 223506):
        pair = [r for r in wave if r["job"]["seed"] == seed]
        assert {r["job"]["head"] for r in pair} == {"box", "tanh"}
        assert len({r["data_sha256"] for r in pair}) == len({r["initialization_sha256"] for r in pair}) == 1
    auxiliary = dict(kind="Source/record extraction, not a new gradient or training run",
                     failed_checkpoint_diagnostics=diagnostics, subsequent_wave=wave,
                     subsequent_wave_passes=sum(r["gate"] for r in wave),
                     independence_scope="Five new paired seeds 223501-223505, disjoint from original 223001-223005; heads share data/init per seed; not ten independent datasets",
                     original_G1_unchanged="10/11; original failed deep configuration remains 3/5",
                     limits="Checkpoint training-domain diagnostic does not establish validation saturation, historical gradients, irreversible trapping or support-wide saturation.")
    (OUT / "synthetic_auxiliary_roster.json").write_text(json.dumps(auxiliary, indent=2) + "\n")
    path = RESULTS / "score_controls_attempt001/summary.json"
    control = json.loads(path.read_text())
    assert sha(CODE / "score_controls.py") == control["sources"]["score_controls.py"]
    last = max(row["step"] for row in control["gradient_stress"])
    protocol = dict(kind="Source-based documentation of existing score stress test; no rerun",
                    source=str(path.relative_to(ROOT)), source_sha256=sha(path), producer=control["sources"],
                    n=10, beta=.3, R="28/3", Lambda="log(28/3)/2", ball_radius="sqrt(2)*Lambda",
                    posterior=[.45, .49] + [.0075] * 8,
                    initial_raw="(4*Lambda,-4*Lambda,...,-4*Lambda)",
                    initial_feasible="Box clip / ball projection of same raw initialization",
                    updates=20000, step_size=2, separate_parameter_moving_terms=False,
                    methods="Raw clip/tanh gradient steps with their Jacobians; direct box/ball projected gradient steps",
                    stopping="Fixed 20000 updates; saved steps 0,1,10,100,1000,20000",
                    zero_convention="oracles.fw_gap returns max(0, FP64 dot-plus-support); zero is reported floating arithmetic, not an outward-rounded exact certificate",
                    PGD_asserted_tolerance=1e-9,
                    endpoints=[r for r in control["gradient_stress"] if r["step"] == last],
                    inspection="score_controls.py lines 48-65; oracles.py lines 21-24",
                    limits="Constructed different domains/Jacobians, not a general optimizer/head ranking")
    (OUT / "score_stress_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(dict(checkpoints=len(diagnostics), wave_models=len(wave), wave_passes=auxiliary["subsequent_wave_passes"],
                          training_diagnostic_norms=[r["full_parameter_Linf"] for r in diagnostics],
                          strict_margins=[r["minimum_distance_beyond_clip"] for r in diagnostics], stress_endpoints=len(protocol["endpoints"]))))


if __name__ == "__main__":
    main()

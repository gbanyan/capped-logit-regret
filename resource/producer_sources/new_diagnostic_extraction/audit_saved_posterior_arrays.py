"""New numerical audit of existing arrays, not a historical check or model run.

Ordinary NumPy FP64 reductions are not outward-rounded certificates. Missing
per-model arrays remain explicit; no data generation or inference is performed.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "response_draft"


def main():
    source = OUT / "saved_oracle_rows.csv"
    entries = list(csv.DictReader(source.open()))
    rows, missing = [], []
    for index, entry in enumerate(entries):
        if entry["cohort"] == "image":
            if entry["split"] == "official":
                path = (ROOT / entry["source"]).parent / "test.npz"
            else:
                phase, job = entry["job"].split("__", 1)
                path = ROOT / "results/capped_logits_completion_2026-10-22/phases" / phase / job / (entry["split"] + ".npz")
        else:
            path = (ROOT / entry["source"]).parent / (entry["split"] + ".npz")
        if not path.exists():
            missing.append(dict(cohort=entry["cohort"], job=entry["job"], split=entry["split"],
                                reason="No per-model saved posterior array at expected locator", expected=str(path.relative_to(ROOT))))
            continue
        with np.load(path, allow_pickle=False) as archive:
            if "p" not in archive.files:
                missing.append(dict(cohort=entry["cohort"], job=entry["job"], split=entry["split"],
                                    reason="Archive lacks p", expected=str(path.relative_to(ROOT))))
                continue
            p = archive["p"]
            assert p.ndim == 2 and p.dtype == np.float64, (path, p.shape, p.dtype)
            values = dict(cohort=entry["cohort"], job=entry["job"], head=entry["head"], split=entry["split"],
                          examples=p.shape[0], classes=p.shape[1],
                          minimum_p=float(p.min()), maximum_p=float(p.max()),
                          negative_entries=int(np.count_nonzero(p < 0)), zero_entries=int(np.count_nonzero(p == 0)),
                          nonfinite_entries=int(np.count_nonzero(~np.isfinite(p))),
                          maximum_row_mass_error_FP64=float(np.max(np.abs(p.sum(axis=1) - 1))),
                          source=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if "oracle_gap" in archive.files:
                values["array_maximum_saved_FW_gap"] = float(archive["oracle_gap"].max(initial=0))
                values["absolute_gap_summary_difference"] = abs(values["array_maximum_saved_FW_gap"] - float(entry["maximum_FW_gap"]))
            if entry["head"] in ("box", "tanh", "ball") and "oracle_scores" in archive.files:
                payload = json.loads((ROOT / entry["source"]).read_text())
                model = payload.get("source_model_record", payload)
                ratio = model["meta"]["ratio"]
                cap = np.log(ratio) / 2
                z = archive["oracle_scores"]
                assert z.shape == p.shape and np.all(np.isfinite(z)), path
                q = np.exp(z - z.max(axis=1, keepdims=True))
                q /= q.sum(axis=1, keepdims=True)
                # Gradient of the row-mass-weighted CE, respecting stored sums.
                gradient = p.sum(axis=1, keepdims=True) * q - p
                if entry["head"] == "ball":
                    budget = np.sqrt(2) * cap
                    violation = np.linalg.norm(z, axis=1) - budget
                    gap = (gradient * z).sum(axis=1) + budget * np.linalg.norm(gradient, axis=1)
                else:
                    violation = np.max(np.abs(z), axis=1) - cap
                    gap = (gradient * z).sum(axis=1) + cap * np.abs(gradient).sum(axis=1)
                values["maximum_oracle_feasibility_violation_FP64"] = float(violation.max(initial=0))
                values["maximum_recomputed_weighted_FW_gap_FP64"] = float(gap.max(initial=0))
            rows.append(values)
        if (index + 1) % 500 == 0:
            print(f"Inspected {index + 1}/{len(entries)} scalar-record array locators", flush=True)
    fields = list(rows[0]) + ["array_maximum_saved_FW_gap", "absolute_gap_summary_difference",
                             "maximum_oracle_feasibility_violation_FP64", "maximum_recomputed_weighted_FW_gap_FP64"]
    fields = list(dict.fromkeys(fields))
    with (OUT / "posterior_array_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = dict(
        kind="New 2026-10-23 FP64 numerical verification of existing stored arrays; no model/solver execution",
        numpy=np.__version__, source_baseline="d1bef6621301666371c8367430479518f1278925",
        scalar_inventory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        array_records=len(rows), missing_array_records=len(missing),
        covered_by_cohort_split=dict(Counter(r["cohort"] + "/" + r["split"] for r in rows)),
        maximum_row_mass_error_FP64=max(r["maximum_row_mass_error_FP64"] for r in rows),
        minimum_p=min(r["minimum_p"] for r in rows),
        negative_entries=sum(r["negative_entries"] for r in rows), nonfinite_entries=sum(r["nonfinite_entries"] for r in rows),
        maximum_saved_gap_summary_difference=max(r.get("absolute_gap_summary_difference", 0) for r in rows),
        constrained_array_records=sum("maximum_oracle_feasibility_violation_FP64" in r for r in rows),
        maximum_oracle_feasibility_violation_FP64=max(r.get("maximum_oracle_feasibility_violation_FP64", 0) for r in rows),
        maximum_recomputed_weighted_FW_gap_FP64=max(r.get("maximum_recomputed_weighted_FW_gap_FP64", 0) for r in rows),
        missing=missing,
        limitations=["Ordinary FP64 reductions, not outward-rounded real arithmetic.",
                     "Repeated model arrays may share posteriors; example counts are not independent samples.",
                     "Zero coordinates are valid for non-full-support synthetic laws; strict positivity is not asserted universally.",
                     "Absent arrays are unverified, even if scalar diagnostics exist.",
                     "This new audit does not establish chronology or exact-law generator fidelity."],
        output_sha256=hashlib.sha256((OUT / "posterior_array_audit.csv").read_bytes()).hexdigest())
    (OUT / "posterior_array_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("missing", "limitations")}))


if __name__ == "__main__":
    main()

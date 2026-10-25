"""Extract existing scalar diagnostics; no solver, inference or training runs.

Outputs are an author-response draft outside the frozen review submission.
Missing fields stay missing. FP32 candidate adjustment is not oracle feasibility.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results/capped_logits_completion_2026-10-22"
PAPER = ROOT / "manuscript/machine_learning"
OUT = HERE / "response_draft"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows, missing = [], []

    def append(path, payload, cohort, job, head, split, metrics, pointer):
        gap = metrics.get("oracle_max_FW_gap")
        if gap is None:
            missing.append(dict(cohort=cohort, job=job, head=head, split=split,
                                reason="No saved oracle_max_FW_gap", source=str(path.relative_to(ROOT))))
            return
        rows.append(dict(cohort=cohort, job=job, head=head, split=split,
                         maximum_FW_gap=gap, candidate_FP32_adjustment=metrics.get("FP32_feasibility_adjustment", ""),
                         source=str(path.relative_to(ROOT)), source_sha256=sha(path),
                         json_pointer=pointer + "/oracle_max_FW_gap",
                         producer_sha256=payload.get("source_sha256", "")))

    image_rows = list(csv.DictReader((PAPER / "data/all648.csv").open()))
    image_known = [r for r in image_rows if r["official_mean_D"] != ""]
    for row in image_known:
        job = row["phase"] + "__" + row["job"]
        path = RESULTS / "phases/official_test_evaluation_v1" / job / "complete.json"
        payload = json.loads(path.read_text())
        append(path, payload, "image", job, row["head"], "official", payload["metrics"], "/metrics")
        original = payload["source_model_record"]
        for split in ("training", "validation"):
            append(path, payload, "image", job, row["head"], split, original.get(split, {}),
                   "/source_model_record/" + split)

    index = defaultdict(list)
    for path in (RESULTS / "phases").glob("*/*/complete.json"):
        index[path.parent.name].append(path)
    for path in (ROOT / "results/capped_logits_mechanism_2026-10-22/jobs").glob("*/complete.json"):
        index[path.parent.name].append(path)
    summary_path = RESULTS / "synthetic_confirmation_summary_attempt001/summary.json"
    synthetic = json.loads(summary_path.read_text())["rows"]
    for row in synthetic:
        matches = []
        for path in index[row["job"]]:
            payload = json.loads(path.read_text())
            if all(payload.get(k) == row[k] for k in ("data_sha256", "initialization_sha256")):
                matches.append((path, payload))
        if len(matches) != 1:
            missing.append(dict(cohort="synthetic", job=row["job"], head=row["head"],
                                reason="No unique saved-record match", matches=len(matches)))
            continue
        path, payload = matches[0]
        for split in ("training", "validation"):
            append(path, payload, "synthetic", row["job"], row["head"], split,
                   payload.get(split, {}), "/" + split)

    fields = ["cohort", "job", "head", "split", "maximum_FW_gap", "candidate_FP32_adjustment",
              "source", "source_sha256", "json_pointer", "producer_sha256"]
    with (OUT / "saved_oracle_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort"], row["head"], row["split"])].append(row["maximum_FW_gap"])
    result = dict(
        kind="Extraction of existing saved scalar records; no numerical solver or model run",
        source_baseline="d1bef6621301666371c8367430479518f1278925",
        inputs=dict(image_slots=len(image_rows), known_posterior_image_slots=len(image_known),
                    synthetic_models=len(synthetic), synthetic_summary_sha256=sha(summary_path)),
        groups=[dict(cohort=k[0], head=k[1], split=k[2], saved_records=len(v), maximum_FW_gap=max(v))
                for k, v in sorted(groups.items())],
        missing=missing,
        limitations=["Missing posterior mass/positivity extrema are not invented.",
                     "A saved scalar is reported execution evidence, not independent solver verification.",
                     "Candidate FP32 adjustment does not measure oracle feasibility.",
                     "Records sharing inputs/teachers are not independent replications.",
                     "Synthetic records lacking explicit producer hash retain source-record hash and baseline locator."],
        output_sha256=sha(OUT / "saved_oracle_rows.csv"))
    (OUT / "saved_oracle_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(records=len(rows), missing=len(missing),
                          missing_reasons=dict(Counter(r["reason"] for r in missing)),
                          maximum_saved_FW_gap=max(r["maximum_FW_gap"] for r in rows))))


if __name__ == "__main__":
    main()

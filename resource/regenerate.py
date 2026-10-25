"""Rebuild saved-data tables/figures; no models or neural fitting."""
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parent
expected = json.loads((root / "expected_asset_provenance.json").read_text())
for item in expected["inputs"]:
    if not item["path"].startswith("results/"): continue
    assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"]
subprocess.run([sys.executable, str(root / "manuscript/prepare/prepare_assets.py")], check=True, cwd=root)
changed = []
for item in expected["outputs"]:
    path = root / "manuscript/prepare" / item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        changed.append(item["path"])
report = {"status": "PASS" if not changed else "OUTPUT_DIFFERENCES", "outputs": len(expected["outputs"]), "differences": changed, "scientific_inputs_verified": sum(i["path"].startswith("results/") for i in expected["inputs"]), "scope": "Saved-data regeneration; no training or inference; internal provenance notes omitted"}
(root / "regeneration_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
if changed: raise SystemExit(1)

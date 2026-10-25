Online Resource 1: public saved-result resource
Sharp Residual Classification Regret for Independently Capped Softmax Logits
Jing-Rung Huang and Wen-Hsiang Lu, National Cheng Kung University
Correspondence: p78084063@mail.ncku.edu.tw

From this extracted ZIP, use Python 3.11+:
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.4.4 matplotlib==3.10.9
.venv/bin/python regenerate.py
.venv/bin/python producer_sources/new_diagnostic_extraction/reporting_diagnostics.py --resource-root .

The first script verifies eight frozen inputs and all 17 generated output hashes.
The second regenerates three descriptive seed-summary CSVs.
The mathematical checker needs only Python 3.9+ standard library:
python3 -I -S theory_checks/verify_counterexample.py --output theory_result.json

All 648 image endpoint rows, 980 synthetic endpoints and 30 raw-affine interval
records are included, preserving failed gates and all cohorts. There are 180
derived deployments among 648 slots. Selected historical producers are supplied
for inspection, with original environment/path assumptions; they are not a
turnkey neural training pipeline. No network fitting, checkpoint inference or
extended-precision interval-solver replay is performed by regenerate.py.
The full trained-model/array corpus and external image archives are not included.
PUBLIC_RELEASE.json traces metadata updates and four administrative origin
redactions. Numerical inputs, producers and expected outputs remain unchanged.
See LICENSES.md: author-written code MIT; author-owned results/docs CC BY 4.0.
See SCIENTIFIC_EVIDENCE_SCOPE.md and ADDITIONAL_EVIDENCE.md for original provenance.

For the single-command fresh-directory runner and file manifest, use the full
repository or its matching capped-logit-regret-v1.0.0.zip release asset:
https://github.com/gbanyan/capped-logit-regret

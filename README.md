# Capped-logit regret: reproducibility resource

**Sharp Residual Classification Regret for Independently Capped Softmax Logits**  
Jing-Rung Huang and Wen-Hsiang Lu, National Cheng Kung University.

This release supplies the standalone mathematical checks and the complete
saved-result package used to regenerate the manuscript's 17 generated
tables, figures and data files. It includes all 648 image endpoint rows
(including 180 derived deployments), 980 synthetic endpoints, 30 raw-affine
interval records, scientific producer sources, protocols and provenance.
All cohorts and failed gates are retained.

## Quick start

Python 3.11+ is required for the pinned saved-result workflow. Only NumPy and
Matplotlib are needed; no GPU, Torch, private repository or model download
is needed. The mathematical checker alone needs Python 3.9+ standard library.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
```

Verified with Python 3.14.9, NumPy 2.4.4 and
Matplotlib 3.10.9 on macOS. Mathematical checks were also verified with
Python 3.15.1 without site packages. Other plotting versions may change hashes.

The runner verifies every supplied file's SHA-256, extracts the resource to
a new temporary directory, runs the theory checker, rebuilds the 17 original
outputs, compares their recorded hashes, and regenerates three descriptive
seed-summary CSVs. It writes a report in `verification/last_run.json` and
keeps the generated output directory named in that report. Failure returns
a nonzero exit code. To check only the mathematics without dependencies:

```sh
python3 -I -S resource/theory_checks/verify_counterexample.py --output theory_result.json
```

## Contents and scope

* `resource/`: scientific sources, inputs, endpoint tables and diagnostics.
* `reproduce.py`: fresh-directory verification and regeneration entry point.
* `MANIFEST.json`: hashes for all distributed files except itself.
* `resource/PUBLIC_RELEASE.json`: changes from the frozen journal resource.
* GitHub Release assets: the matching `ESM_1.zip` and a self-contained
  `capped-logit-regret-v1.0.0.zip` with this entire resource.

The 17-output replay is complete **from saved summaries**. It does not
retrain networks, rerun neural inference, or independently replay the
extended-precision conditional interval solver. Selected historical producer
sources are supplied for inspection; they retain their original environment
and path assumptions and are not a turnkey full-training pipeline.
The full trained-model/array corpus (24.78 GB summed over paths) is not
included in this release; its separate public deposit has not yet occurred.
The external Fashion-MNIST/CIFAR-100 image archives are not redistributed.
Do not interpret successful numerical tests as a formal proof or claim of
exhaustive novelty. See `resource/SCIENTIFIC_EVIDENCE_SCOPE.md`.

Original scientific inputs, numerical producers and expected outputs are
byte-preserved. Four private-origin strings in administrative source/audit
records are redacted and traced in `PUBLIC_RELEASE.json`. Historical paths,
source hashes and execution records identify original runs; they are not
commands to resume old queues or claims of new execution.

## License and citation

Copyright 2026 Jing-Rung Huang and Wen-Hsiang Lu.
Author-written Python code is MIT licensed (`LICENSE-MIT.txt`).
Author-owned results, metadata and documentation are CC BY 4.0
(`LICENSE-CC-BY-4.0.txt`). These grants do not relicense cited third-party
works, external datasets or dependencies. `CITATION.cff` describes this
software resource; the manuscript is not claimed to have been published.

Repository: https://github.com/gbanyan/capped-logit-regret

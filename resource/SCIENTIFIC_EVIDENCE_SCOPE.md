# Scientific source and saved-data reporting additions, 25 October 2026

Original endpoint arrays, gates, thresholds, model fits and the seventeen
saved-data generator outputs are unchanged. This extension adds existing
scientific producers and original execution/record closure, plus explicitly new
read-only summaries. It contains no earlier AI review reports or adjudications.

Protocol records preserve exact original byte ranges. RESOURCE_ADDITIONS.json
records the whole-source SHA256 and byte offsets. The teacher-centering and
synthetic-confirmation records supply whole original scientific protocols; the
other three are excerpts. The metadata identifies when nothing was omitted.
Author/review-process narrative outside the selected ranges is omitted. None of
these records is claimed as public preregistration. Runtime manifests still pin
whole original source files.
Git/launch chronology is recorded evidence, not authentication of every prior run.
git_lineage_commands.json supplies the new local Git object-graph commands and
their exit codes; this is not external authentication of historical timestamps.

diagnostics/natural_seed_dispersion.csv and paired_natural_error.csv use all five
natural-label seeds per head. Error units are percent/percentage points; CE is
nats. SD is sample SD. SE is descriptive SD/sqrt(5), with fixed data and teachers;
no p-values or population confidence claim is made. affine_seed_dispersion.csv
retains all twenty-four head/arm cells. These are new summaries of saved rows,
not new experiments. Regenerate them using the included reporting_diagnostics.py.

warm_retention.json checks original epoch-zero/final histories, not a new
checkpoint forward. raw_feature_precision.json records a new ordinary-FP64
std computation on stored original FP32 features. Its source NPZ remains in the
full corpus, not this lightweight resource. It is not an extended-precision
certificate. The .1-floored training normalizer is a separate stored quantity.
The final conditional intervals have not been independently replayed here.

The original primary serialization audit, baseline/teacher records, ordered
queues, launch/source manifests and CIFAR reader/identity are now supplied.
The external CIFAR tarball was not part of the cold-retrieved Git payload.
The GPU container mounts a read-only host venv; its package inventory is distinct
from the container image digest and from the CPU serialization-audit environment.

The original README's internal-protocol exclusion refers to author/review notes;
the scientific excerpts above are intentional new evidence. The existing source
and output generator is unchanged. Run regenerate.py for its seventeen outputs.
Full model/array access, public archive location and reuse terms remain separate;
no public deposit, license grant or on-request access promise is created.


Public-release update: https://github.com/gbanyan/capped-logit-regret, v1.0.0. The lightweight scientific
resource is now distributed publicly under LICENSES.md. Earlier local-packaging
statements above record their preparation-time status. Full model/array corpus
access remains separate. See PUBLIC_RELEASE.json for the exact public changes.

# Source/version reconciliation, 25 October 2026

The comparison is limited to the retrieved primary-source files listed below.
`source_access.json` records actual URLs, retrieval UTC, status, content type,
bytes and hashes. Public source PDFs are kept only in the local ignored work
directory; no third-party full paper is redistributed in the minimal checker.

| Source | Retrieved SHA-256 | Compared portions |
|---|---|---|
| NeurIPS 2023 proceedings | `04f4e07cf701a6855b08115108ced54de568063d9b566a31f2a707ff82aafd68` | Section4.2 Theorems5/6; E.1 PDFpp28–29; E.2 pp30–31 |
| Current author-hosted `chcb.pdf` | `841b8af7cb78d6261d361afcc2c8a2e78597012ba6fdc9c883ac4ea0da6dda2d` | All39 pages compared by normalized text; relevant pages read |
| Zheng2023 ICML proceedings | `bcdc6a7f3db855531f8dac5ffd8e765ea6564fb4eacf5dd2dacf8b42887e9a7b` | H_lin definition p3; Theorem3.3 p4; proof E.4 |
| Enhanced H-Consistency Bounds, ALT2025 proceedings | `a680d00500a1b72c6627b9008b206e1583b0bff604ea3fa1c785bf0c854205d6` | Theorems1/2/4/5/9, Table3, relevant conditional premises |
| Enhanced H-Consistency Bounds, author arXiv v2 | `d99c8080d271eb1e3888317e25a09ad3f52de294cfdf6cf12acac2f98678363d` | Corresponding framework/completeness/power-bound premises; not labelled byte-identical to proceedings |

The current author-hosted NeurIPS copy differs at file-byte level but all39 pages
have identical whitespace-normalized extracted text to the proceedings copy.
`version_text_comparison.json` records this actual comparison. It is not a pixel
identity, a claim about every historical revision, or proof that no erratum exists.

The OpenReview conference record is `nI7EmXq2PL`. The bibliographic entry
`F3BYo1fgD5` displays a May2025 metadata modification in the indexed public view;
that timestamp alone does not establish a theorem/PDF revision. Direct retrieval
of both forum entries redirected to browser verification. The conference PDF
and both API generations returned HTTP403 for the attempted note IDs. The local
browser-control inventory was empty, so no browser retrieval or CAPTCHA action
occurred. No author/editor was contacted. Consequently the accessible text
comparison is completed, while full OpenReview revision/erratum history remains
unresolved. The manuscript states that limit and does not use it as evidence of
absence of correction.

The precise NeurIPS feasibility obstruction is the asserted family in E.1 p29,
which E.2 then invokes: its marginal coordinate interval can permit a probability
transfer outside the box's joint ratio polytope. The new strict rational example
and its exact arithmetic are in `theory_checks/verify_counterexample.py`.
This is separate from establishing the manuscript's universal sharp residual.

For Zheng2023 the original (.5,.4,.1) posterior is outside the displayed
Theorem3.3 spread condition. The different posterior (17/40,2/5,7/40) satisfies
both displayed premises but contradicts the ordinary-gap bound on the singleton
x=0; its KKT, exact excess and small-error check are in the standalone checker.
The theorem does not invoke the feature-nondegeneracy assumption of the separate
naive-Bayes analysis. A second .24-spread variant checks that equality at the
spread threshold is not essential. This finding targets the displayed restricted
bound, not its entropy-relative Bayes-calibration counterpart, not all other
theorems in that paper, and not unobserved later corrections. In E.4, LemmaE.5
uses an unsupported upper bound on the exponential normalizer in the reverse
implication from spread to unconstrained-optimum feasibility; this manuscript
uses the direct counterexample rather than relying on that claimed equivalence.

ALT2025 provides conditional-to-population tools conditional on pointwise
inequalities. The constrained-loss application assumes completeness; the
logistic examples in Table3 also assume symmetric, complete classes. Theorem9
assumes an unweighted zero-residual conditional power bound and low-noise
conditions. Its generic weighted framework is not contradicted merely by a
sequence when predictor-dependent factors may diverge. These compared
statements do not establish the finite-box sharp constant. This is a
statement-level comparison, not exhaustive field-wide priority certification.

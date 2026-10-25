# Prospective centered-logit representation and native score-field input

This is the concrete refinement of the unlaunched disjoint_features proposal,
before any teacherA/B feature output. It preserves all original raw-teacher
pilot and full-data repair results. All A/B/validation partition,CDF,normalizer,
posterior and no-test rules of disjoint_features_protocol remain in force.

Independence correction BEFORE ANY new disjoint feature output:the brightness
CDF used to DEFINE p must be calibrated on teacher-only A,not student B.
Otherwise the target law itself still depends on student inputs after fixing
the teacher. This supersedes the earlier unlaunched B-CDF proposal. Store the
CDF-fit A-index hash. Student logit normalization remains fit on B as a learned
preprocessor,but it does not define target p. Teacher weights,centering,
posterior CDF and law are all fixed from A before student training on B.

Remove the irrelevant common logit gauge from each fixed halfA teacher by
centering its FINAL AFFINE weights over output rows and centering its bias.
Do so in FP64 after loading the exact FP32 coefficients. The native architecture
is unchanged;for ideal arithmetic this subtracts mean(r) from all raw scores,
so natural probabilities and argmax are unchanged. No B or validation outcomes
choose this transformation. It addresses a standard softmax gauge nuisance in
the failed raw-logit readout and supplies a nontrivial centered score field for
the separately labeled KKT law. It is not a new clipping method or retrospective
relabelling of original-score outcomes.

Save the actual centered FP64 native checkpoint. Infer its logits in FP64 on
the training-source images;B only supplies student logit normalization,
teacherA supplies pixel normalization and posterior brightnessCDF. Compare validation CPU
scores to ROW-CENTERED original GPU teacher logits (<=1e-3),record probability
and decision differences. These are numerical comparisons,not claimed bitwise
softmax invariance. The exact saved centered checkpoint defines the later
controlled p function;no oracle p is a deployed student input.

Mixed teacher-probability law remains softmax(logits) mixed with cyclic class
roles and A-calibrated u. Centering is mathematically softmax invariant;
any finite-precision function change is explicit. The KKT law is DIFFERENT and
depends on the centered raw scores,so it gets distinct law IDs and source pins.
Its cap sweep changes p;fixed-p comparisons must be labeled separately.

All raw-scoreKKT135 controls remain reported,including small CIFAR regret.
Do not select only an adverse cap or hide a gauge. Before actual student output
freeze the full-B4x512 development and paired confirmation sources/queues.
No official test decoding,seed exclusion,gate reduction or shutdown here.

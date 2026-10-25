Natural60:all15 raw and45 bounded trained models,final prescribed checkpoint,
no best-validation/test selection. CIFAR100 uses the last50 original official
test indices within EACH fine class(5000);the old frozen program used the
first50. FashionMNIST uses all official10000. Hash the original test members
and selected index bytes. The selection rule is fixed before test decoding.
Use each model's saved training-only pixel mean/std,no fitted test statistics,
no augmentation,FP32,noAMP/noTF32,evaluation mode. Report natural CE and error,
never natural Bayes regret. Training uses no test labels or outcomes.

Controlled image laws use the EXACT saved centered teacherA FP64 checkpoint,
A-only pixel normalization and A-calibrated CDF from the clean feature pack.
Compute logits on test pixels in CPU FP64 with that checkpoint;interpolate
brightness with the stored A-CDF without recalibration. Student logit mean/std
remain the stored B values. For each job,recreate its mixed/KKT posterior and
output permutation with the frozen law parameters. Known p is a metric/target
reference,never an additional random pixel student's forward input. Keep
artificial-posterior statements separate from the natural-label evaluation.

Evaluate every120 confirmation readout and168 full-pixel trained models with
fresh checkpoint loads. Evaluate the derived capacity witnesses separately
without counting them as new seeds. Save scores,raw scores,p,oracle scores and
certified oracle gaps,selected image indices,metrics,source/model/input hashes,
runtime and exact fresh-load replay comparisons. Preserve material numerical
differences between FP32 students and the FP64 target generator. Recompute
saved endpoint metrics independently during archival. No retraining or threshold
selection on these test outcomes;new exploratory repairs need a separate set
and an explicit exploratory label.

Summaries use the five paired trained seeds as the replication unit,not images,
heads,tie-rule reevaluations or shared teachers as extra independent seeds.
Report seed values and descriptive paired mean differences with small-sample
uncertainty;do not assert dataset-universal confidence from three fixed teachers.

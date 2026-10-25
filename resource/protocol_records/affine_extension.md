## Finite models and immutable inputs

Use the SAME three centered teacherA checkpoints,disjoint studentB/validation,
beta.3,kappa.95 native KKT targets and five permutations/seeds227001–227005 as
native60. Teacher,normalization and CDF fit A only;no official data is read.
All backbone/BN/dropout modules stay in eval mode and all non-final-affine
state bytes stay fixed. For FashionCNN only output[-1] is trainable;its first
output affine/ReLU/dropout belong to the frozen representation.

Cache the final-affine input on B and validation separately in original order,
batch128 with exact final batch sizes. Source/input/state pins and strict
capture/fresh-model/isolated-affine checks follow affine_feature_protocol.md.
Compute feature mean/std from B only in FP64,std floored at.1. Shared training
coordinates are (phi-mean)/std cast to FP32. All heads/arms use identical
features,targets,initial head and data order. The underlying real affine class
is not replaced by a bounded parameter box. Floating folding differences are
reported;the learning gate ALWAYS uses actual whole-pixel native forwards.

Two separate phases,each3architectures x4heads x5seeds=60 trained models:

-pixel_affine_primary_v1:100epochs,SGDlr.1/momentum.9 for ResNet,Adam.001 for
 Fashion,zero weight decay,cosine schedule eta_min=initial_lr*.001.
-pixel_affine_secondary_v1:identical,only initial_lr divided by10.

Both optimize final-affine coordinates only,soft CE with FP32 targets. Initial
normalized-coordinate Linear is reset once with torch.manual_seed(seed),then
cloned identically for each head/arm. Data order uses its own CUDA generator
seed+11,batches128. No AMP/TF32/augmentation/early stopping or hyperparameter
search. History may evaluate cached head proxies;final gates/arrays are from
actual native pixel evaluation on ALL B and validation with frozen body.

Physical deployment folds W_native=W_normalized/std and
b_native=b_normalized-W_native*mean into the native final affine,FP64 arithmetic
then one FP32 cast. Save normalized head state and full deployed model. Verify
the exact non-head state hash before/after. Report B/validation cached-versus-
deployed raw/score/CE/D/decision deltas,never assume the two machine functions
are bitwise equal. Whole-model final-helper replay itself remains BITWISE.

Three15-model zero-step phases,all architectures/seeds,not selected by results:

-pixel_affine_teacher_reference_v1:actual FP32 teacher-affine point,permuted
 into the same label law. Keep all15 models/arrays,not warm epoch0 scalars.
-pixel_affine_raw_head_transfer_v1:all15 PRIMARY raw checkpoints with box head.
-pixel_affine_tanh_head_transfer_v1:all15 PRIMARY tanh checkpoints with box head.

Transfer model bytes stay identical to their parent;zero optimizer updates.
These are deployments,not projected-gradient training. Teacher references are
known-function points,not learned successes. The primary/secondary phases are
not120 independent dataset/seed replications;each cell has the same5 seeds.

New30 box endpoints(primary15+secondary15) receive full-B deployed checkpoint
gradient/saturation diagnostics with the final-affine trainable mask. Include
passing and failed models. Report native affine-coordinate gradients and the
declared coordinate transformation;do not label these all-native-parameter or
historical stochastic-optimizer gradients. Old69 diagnostics remain unchanged.

## Three distinct questions and gates

Q1 learning:unchanged per-model meanD<=1e-3 AND nearcoverage>=.5. A group labelled
successful needs>=4/5;primary and secondary results stay separate. Never replace
a failed primary gate with a successful secondary gate or derived copy.

Q2 adverse selection:report signed regret difference between actual deployed
box decisions and raw-order decisions on the SAME actual near set,including
zero/negative changes. Also report clipping-created errors when raw is Bayes,
ties and posterior margins. A near-optimal model plus an upper-bound audit
alone is NOT evidence of harmful selection. No claim SGD must find adverse
minimizers. All box/tanh endpoints require the accepted pointwise bound audit.

Q3 raw optimization:report an empirical TRAIN-B affine objective interval,
its numerical allowances and each raw learner's objective gap upper bound.
Reference width<=1e-6 nats is the declared tight-reference criterion;GPU raw
adequacy needs conservative U_gpu-L<=1e-4. Any unresolved rank/error bound or
wide interval is UNKNOWN optimality,not a PASS. This never erases valid Q1/Q2
outcomes. No validation/test or whole-native-class optimum is certified here.

Architecture-arm qualification for Q3 requires ALL5 raw seeds adequate. Primary
remains the primary Q1/Q2 arm always. If primary fails Q3 qualification but the
secondary qualifies,report the entire secondary architecture arm as the Q3
comparison/sensitivity arm for ALL heads. Otherwise report Q3 unknown or use
the qualified primary. No per-head/seed/lr cherry-picking or further recipe search.

The actual teacher reference must satisfy the tighter input preflight
meanD<=1e-6/coverage>=.99 on B and validation. If it fails,preserve/review the
reference;do not infer no FP32 model could pass and do not weaken learning gates.
Old random-pixel vs new frozen-body models differ in pretraining and trainable
class;that contrast alone is not a single-factor causal isolation.

## Raw affine reference:unbounded class,three canonical problems

There are THREE canonical training objectives,one per architecture. Fifteen
seed problems are exact class-column permutations,verified on stored targets
and transported coefficients. They are not15 independent solver replications.
Reference fitting never uses validation/test targets. Known-p targets are
stored FP64 T_ij>0 with rowmass s_i=sum_j T_ij,not silently renormalized to alter
the objective. Let a=min(T),bar_s=mean(s_i),nclasses>=2. Let Z be the full-rank
training design span including bias,or a VERIFIED invertible coordinate change.
For centered Theta1=0 define F=mean[s_i*logsumexp(Z_i Theta)-T_i*(Z_i Theta)].

If lambda<=lambda_min(Z.T Z/N),h>=max_i||Z_i||,then
F>=a*sqrt(n/(n-1))*lambda*||Theta||F/h. Therefore the unconstrained optimum
exists in the centered quotient and has norm at most
R0=h*bar_s*log(n)/(a*sqrt(n/(n-1))*lambda). Any attained upper objective can
tighten bar_s*log(n). Row-minimum weighted Gram bounds may tighten a*lambda.
This radius bounds an optimum;it does NOT constrain/change the trained class.

For ANY Q>=0 with exact mathematical rowmass s_i,entropy conjugacy gives
L(Q)=mean[-sum Q_ij log(Q_ij/s_i)]-R0*||Z.T(Q-T)/N||F <= min F.
Combine with the unconditional weighted-entropy lower bound from T. Report
[L,U] and training affine-class D-floor interval relative to that entropy.
Primal stationarity or solver.success is never a gap certificate.

Use full-support raw CE in FP64,L-BFGS as a solver(max2000iterations,max10000
function evaluations,maxcor25,maxls40,gtol1e-12,ftol2e-15). Zero initial
coefficients,center class gauge,all data B. Record termination and attained
point regardless of certificate quality. Preconditioning is B-design-only.
Do not silently truncate near-null singular directions. Exact zero B columns
may be removed with recorded span equivalence;otherwise unresolved rank means
UNKNOWN certificate. No smaller bounded parameter class is substituted.

Fixed dual candidates:T;weighted softmax at attained point;one and two passes
of actual-Gram null-residual projection,shrunk toward T to stay positive and
mathematically rescaled to exact rowmass. Approximate whitening is not assumed
orthonormal;use its actual Gram solve and record all defects. Taking the maximum
of valid lower bounds is a numerical audit,not trained-model/recipe selection.

Report verified Gram lower bounds,feature norm upper bounds,target positivity,
radius,entropy/rowmass/moment/upper-objective error allowances and actual gap.
Propagate operation-count error bounds;do not copy a blanket1e-13 allowance.
Use independent extended precision where available and high-precision spot
checks as checks,not proofs. State IEEE/libm/error-model assumptions. A failed
allowance/rank guard leaves an attained reference with UNKNOWN optimality.
No universal formal machine-arithmetic certificate is claimed. Shared-class
gaps of raw/ball/tanh in the FULL native network remain UNKNOWN.


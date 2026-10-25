# Prospectively frozen neural confirmation and generality matrix v1

Not a manuscript change or full-program completion. Preserves old48 and current
development8. Development222400passed its fixed gate:boxCE meanD4.2266e-7,
coverage.999756,regret.067865;rawCE3.3779e-5,ballCE6.3980e-6,tanhCE1.0087e-7.
Every checkpoint replay exact. This selected the400-epoch recipe,not new seeds.
TanhFP32ties require precision controls;no universal performance dominance.

## Complete finite matrix

Core7configs:(4,.1),(4,.3),(10,.1),(10,.3),(10,.6),(40,.3),(40,.6).
Both realizable and mixed laws x4heads(raw,box,tanh,ball) x2objectives(CE,oracle
distillation) x5paired NEW seeds223001–223005,MLP2x256 =560. Binaryn2,R3/R30,
single smooth binary law,same heads/objectives/seeds=80. Binaryraw duplicate
controls are deterministic copies acrossR,not ten independent seeds.

Depth:representative(10,.3),(40,.6),bothlaws,4heads,2objectives,5coreseeds,
MLP4x512=160. Held-outconfigs(7,.2),(25,.45),bothlaws,4headsCE+raw/box
distillation,NEW223101–223105,MLP2x256=120. Held-outlaw additionally broadens
disjoint-input noise to.15 and gamma sign(t)|t|^2 (separate newly trained tasks,
not unannounced concept shift at evaluation). Smooth overlapping Gaussian
mixture n10,beta.3,4heads,2objectives,NEW223201–223205=40.

Network saturation controls:box/tanhCE n10,beta.3,realizable,5coreseeds,output
bias +/-2cap selected by independent seeded signs=10. They intentionally alter
initialization,remain separately labeled and are not adequacy-gate arms.
Precision controls:box/tanhCE n10,beta.3,realizable,5coreseeds,FP64 training=10.
Initialize same FP32 weights before conversion;this separates arithmetic from
head geometry and does not change the old pilot. Total980 neural arms. Shuffle
execution order seed223400. All remain required;no seed/head outcome exclusions.

## Inputs,posterior and numerical conventions

8192 training and4096 independent heldout validation inputs per task. All runs
400epochs,batch512,Adam lr.002,cosine eta.00002,FP32 unless explicitlyFP64,noAMP,
deterministic torch,2CPU/8GiB/oneGPU. No early stopping,best checkpoint selection
or validation recipe search during this matrix. Dataset generation and class
permutation use separate RNG states (SeedSequence[family,19031] for permutation),
and all heads/objectives of each task share initialization and minibatch order.
BinaryR is an explicit job input. Realizable/mixed formulas as in development;
support-wide realizability derivation retained. No latent-cluster posterior error.

Gaussian law:20 isotropic unit-covariance components inR10,equal weight. Classes
c=0..9,groupg=floor(c/2),sign=2(c mod2)-1. Both component centres have first-five
part6e_g and private coordinate5+g =sign*.5. Component0/1 adds +/-.35 to private
coordinate5+(g+1 mod5). Label permutation fixed independently perseed. Draw x
from this mixture,compute actual p(c|x) by summing BOTH component densities per
class and normalizing over ALL classes. Feed scaled x/6 and one independent
uninformative coordinate to the student. Do not train on hidden sampled labels
or pretend sampled component assignment is observable p. Full Gaussian overlap,
strictly positive posterior,mostly ambiguous paired classes. Centres/scales fixed
before any model outputs;oracle-only development check expects20–60% top-clip
tie prevalence. If not,retain and repair prospectively,not discard the law.
There is no requirement that posterior tail mass equal cap beta.

Save all raw network preactivations and actual headed outputs. Report actual
FP32/64 ties as well as ideal monotone raw-order decisions for tanh/positive
radial ball. FP64tanh can still round;raw order is the mathematical comparator.
For oracle comparisons only,remove head feasibility roundoff in float64,record
correction<2e-6. Analytic box top-clip membership and0/1e-12/1e-10 sensitivity
define tie strata;do not use approximate ball ties to label box geometry. All
conditional excess/certificate calculations retain domain-specific FW errors.
Persist failures,partial model/validation arrays and traceback;continue independent
jobs but phase exits nonzero if any fail. A missing complete.json is never a PASS.

## Fixed assessment criteria and allowed conclusions

G1:for ordinary boxCE on each realizable core/depth/heldout config,meanD<=1e-3
AND fractionD_upper<=1e-3>=.5 in>=4/5seeds. This tests optimization/learnability
where actual shared class has a support-wide zero CE-gap witness. A failure is
retained and needs explicit diagnosis/prospective repair,not a theorem failure
or a retrospective tolerance/width change. Distillation success (meanD<=1e-3)
is an independently labeled attainable-risk control for SAME architecture.
Failed distillation is inconclusive about capacity. Mixed/ball/raw classes do
not inherit the box's zero ordinary CE gap.

H1 exploratory but fixed:near-optimum hard-box runs often attain exact top ties;
conditional disagreement around.5 is expected only on actual learned tie strata
under balanced signs. Compare observed regret to oracle fixed-index/random-tie
expectations on the SAME selected N={D_upper<=1e-3},not an unconditional formula
after selection. Population realizable fixed-tie reference=.225beta;strict
adverse near-minimizer existence is separately proven/checked,not imposed on SGD.
Report strict wrong learned near-minimizers separately from exact/near ties.
No learned adverse-frequency threshold is a mathematical theorem test.

H2:raw/ball usually retain Bayes order near their own optima;compare regret only
on common near sets with>=50% overlap and report actual posterior-margin support.
Do not infer comparative capacity or equal excess from equal natural CE. On
strict-oracle strata,report the actual domain probability gap and certificate
coverage. Integrity criterion:all examples with2D_upper<gap_lower^2 must be Bayes
correct;this is a numerical theorem check,not an independent empirical discovery.

H3:measure hard/tanh saturation and whether stressed initialization recovers;
interpret tanhFP32vsFP64and ideal-order differences as arithmetic/optimization,
not automatic evidence of a distinct mathematical calibration residual.
Binary is a zero-positive-residual control,not identical finite learning curves.
Uniform/deterministic/high-margin/strict near-minimizer point controls already
archived;the finite curve is not claimed sharp. Record first/last-index regret,
random tie expectation,permutation,all histories/checkpoints and exact replay.

Stats:seeds are the replication unit,paired by architecture/config/law. Report
all five endpoints,mean,SD,t4 95% intervals and paired head contrasts. Input
fractions get conditional descriptive counts;do not treat every minibatch/input
as an independent model replication. Heldout configs/law are prospectively
specified external checks,not a claim over all neural networks/distributions.
Official image tests remain sealed. Teacher-image tasks and full R1–R7 final
audit are still required after this matrix;no shutdown at its completion.

Prelaunch local check:DEVELOPMENT seed222550,n4/10/40,2048 inputs each verifies
the explicit CE and Bayes witnesses embedded in the ACTUAL2x256 architecture,
both binary ratios with all domain oracles,and32768 Gaussian-mixture inputs
verify frozen tie prevalence20–60% before training. Confirm exact980 names/group
counts and seed/permutation pairing. This is not a confirmation seed/outcome.
Preserve any failure;no GPU launch until this implementation/law check passes.

# Frozen independent image confirmation and full-pixel recipes

Concrete finite implementation of image_remaining_program.md. Before any
confirmation output,pin the three audited disjoint_features_centered_v1 packs
and actual producer/dependency bytes. Target teachers and brightnessCDF were
fit on A only;all students train on B only. Original validation is held out.
No official test is decoded by these producers. ALL arms/seeds are retained.

The development recipe is already fixed as four512-wide ReLU layers,400epochs,
Adam.002/cosine decay to.00002,batch512,FP32 for feature readouts. Run the full
24-arm development;the confirmation architecture is this same recipe,including
if development fails its unchanged meanD<=1e-3/nearcoverage>=.5 gate. A failed
gate means negative/inconclusive evidence,not permission to select a different
confirmation subset. Any later material recipe repair needs a new protocol.

Confirmation readouts:3teacher architectures x4heads x2objectives x5paired
seeds225001–225005=120,400epochs,all B,CUDA. Separate CE from oracle-score
distillation. The two objectives share the same input/initialization/permutation
recipe. Distillation targets may use known p during training;student forward
receives teacher logits and observable u,never true p.

Full-pixel recipes use the already implemented image_pixel producer and its
12-case actual-CNN/permutation/transplant wiring check. Native B.model_for
backbones are ResNet18,FashionCNN,FashionMLP. teacher_mixed adds the matching
four512-wide readout;image_kkt uses the native backbone alone. The forward of
random students consumes original normalized pixels and observable u only.
No pretrained teacher function is secretly passed as the random model input.
All weights are independently initialized unless explicitly marked warm.

Queue stages,with no outcome-based exclusion:
- pixel_mixed:60 soft-pCE models,seeds226001–226005,100epochs,fourheads.
- pixel_kkt:60 soft-pCE models,seeds227001–227005,100epochs,fourheads,
  beta.3,kappa.95,centered native teacher score field.
- pixel_hard:18 models,box/raw,seeds227101–227103,100epochs,the SAME KKT law,
  one sampled hard label per B image using seed+4141,stored for replay.
- pixel_warm:30 models,box only,seeds227201–227205,10epochs,initialize from
  the centered teacher with the seed's output permutation,paired frozen-vs-train
  BN/dropout statistics. Weights remain trainable;labels and shuffle paired.

All full-pixel training:FP32,noAMP,noTF32,deterministic kernels,noaugmentation,
no weightdecay,batch128 ResNet/batch256 Fashion. ResNet SGD.1,momentum.9;
Fashion Adam.001;cosine schedule down to.001 times initialLR. Warm learning
rates are100x smaller. Conventional random models train their BN/dropout;
the warm pair explicitly isolates that stochastic-layer difference. Train-time
stochastic risk is not automatically identical to deterministic evaluation risk.
Soft-p/hard-label targets and different controlled laws are kept distinct.

Development and confirmation have disjoint seeds. Permutations,tie rules and
evaluation resamplings are not additional independently trained seeds. Report
all per-seed D/regret/gates and training/validation behavior;retain saturated
or failed learning as evidence. Native box KKT class has a support-wide CE
and0–1 witness;native other-head CE gaps and mixed native-class gaps are unknown.
No theorem forces SGD to select a harmful minimizer. Final sealed image tests,
capacity replays,statistical summaries,archive and reviewer audit remain required.

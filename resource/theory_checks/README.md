# Standalone checks for the capped-logit mathematical counterexample

Python 3.9 or later, standard library only. No repository, external data, model,
training, network, NumPy, SciPy or GPU is required. Copy this directory anywhere:

```sh
python3 verify_counterexample.py --output result.json
```

The default performs the following separate checks:

1. Exact rational mass, box-KKT signs and branch conditions for
   p=(1/2,2/5,1/10), q*=(2/5,2/5,1/5), R=2.
2. The mathematical separation certificate at eta=1/1000:
   D(eta) <= eta/2 = 1/2000, while Psi(1/10) >= (1/10)^2/2 = 1/200.
   The latter follows from Psi(0)=Psi'(0)=0 and Psi''(t)=1/(1-t^2)>=1.
   Box KKT proves the stated CE optimum; the perturbed prediction is uniquely
   class2, so its regret is 1/10. The singleton has both ordinary minimizability
   gaps zero. The contradiction does not depend on a numerical optimizer or ties.
3. Two CE evaluations at 80 Decimal digits, across twelve decreasing eta values.
   At eta=.001: D approximately .000100119991996 versus required .00500836684636.
   Decimal evaluation supports the analytic certificate; it is not interval arithmetic.
4. An exact infeasible transfer permitted by the printed E.1 family (PDF p29):
   source q=(.39,.40,.21) is feasible and strictly predicts class2; mu=-.03 lies
   in the printed interval [-.10,.11], but the transformed (.43,.36,.21)
   satisfies all marginal bounds [.2,.5] and violates q1<=2*q3.
5. Two exact full-support variants for Zheng2023's displayed Theorem3.3:
   p=(17/40,2/5,7/40) has spread1/4; p=(21/50,2/5,9/50) has spread.24<1/4.
   At singleton x=0 with B=log(2)/2, both meet its stated posterior/small-error
   premises, yet their strict perturbations violate its square-root bound.
   The check compares 2*D with regret^2 using exact rational upper bounds at
   eta=10^-6. This concerns ordinary minimizability gaps; the corresponding
   entropy-relative Bayes-calibration bound is unaffected.
6. Nine exact rational full-support KKT witnesses with regret=.999*beta and
   matching slope identities. Finite examples do not prove a universal supremum.
7. 5,000 new deterministic floating-point posterior/logit cases (seed20261025),
   checking the affine bridge for every label and the finite bound for every
   exactly maximizing label. The oracle uses scalar clipping/bisection, not a
   numerical optimizer. Sparse posteriors and n=2/3/4/10/40, R=1.01/2/3/30/300
   are included. Absolute tolerance is 1e-10. Tiny negative computed excesses
   within tolerance are recorded and clipped to zero only for the square root.

Use `--samples 0` for just the rational/Decimal witnesses or choose another
`--samples`/`--seed`. Failure raises an exception and returns nonzero. PASS
means these checks passed, not a formal proof or an exhaustive prior-art audit.

The simple contradiction has regret 1/10, whereas beta_3(2)=1/5. It disproves
a zero-residual guarantee; the manuscript's independent upper-bound/sharpness
argument is still needed to establish the exact residual. The comparison is
with the pinned NeurIPS proceedings version, not every unobserved revision.
Public source versions and scope are in the accompanying literature records.
This public release licenses author-written code under MIT and author-owned results under CC BY 4.0; see LICENSES.md.

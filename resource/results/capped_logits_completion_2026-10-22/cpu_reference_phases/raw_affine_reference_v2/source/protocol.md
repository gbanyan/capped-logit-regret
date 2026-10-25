# Proposed conditional arithmetic audit,not a formal interval package

This implements the already reviewed unbounded-class radius/entropy-moment
argument. Only exact constant TRAIN-B columns are absorbed into the bias;no
small singular direction is discarded. X is the original storedFP32 feature
matrix interpreted exactly in real arithmetic,with a bias column. T is the
storedFP64 posterior,not silently normalized. All target mass sums are retained.

Solver coordinates use a B-only standardizer and ALL singular directions when
available. The resulting storedFP64 W is an exact real change of coordinates
ONLY if invertibility is verified. Its inverse residual is bounded by Higham
gamma dot envelopes and a Frobenius norm upper bound. Approximate SVD output
and optimizer success are not accepted rank or optimality certificates.

Numerical repair v2:CPU-reference standard deviations use every positive
nonconstant variance without the .1 floor of neural TRAINING coordinates.
Zero-variance directions are retained by a unit scaling if supplied to a
fixture;production constant removal has its prior exact span proof. No tiny
nonconstant column is removed. Check both inverse_candidate*W and
W*inverse_candidate with separate propagated bounds;EITHER norm<1 proves W
invertible. Physical input units can make just one product badly conditioned.
This changes coordinates of the same unbounded class,not neural recipes.

Let Zhat=fl64(XW),E_Z bound its Frobenius error,and Ghat=fl64(Zhat.T Zhat/N).
Component dot bounds and
||Ghat-G_exact||<=dot_error+(2||Zhat||F E_Z+E_Z^2)/N
give lambda_lower=1-||Ghat-I||F-total_error. Requirelambda_lower>.5 and verified
W invertible. Maxrow||Zhat||+E_Z bounds h. Extra operation-count allowances
cover norms,divisions and scalar arithmetic. If these guards fail,retain the
FULL variable dimension and an attained point,but report rank/optimalityUNKNOWN.

The solver always forms physical coefficients B=fl64(W Theta),then evaluates
X B. The final attained B is storedFP64 and defines an actual member of the
original real affine class regardless of solver approximation. Its upper
objective is reevaluated with extended precision and dot/LSE/target-mass
allowances. No bounded parameter class or numerical-rank truncation is used.

Radius R=h*min(U_zero,U_attained)/(min(T)*sqrt(n/(n-1))*lambda_lower) bounds an
optimum in centered W coordinates. There is no dependence on solver gradient
norm in the claimed gap. The optional weighted-Gram tightening is not used.

Correct coercivity proof:for centered w,let M=max(w). Every w_j>=-(n-1)M,
so summing(M-w_j)(w_j+(n-1)M)>=0 gives||w||^2<=n(n-1)M^2. Therefore
nM>=sqrt(n/(n-1))*||w||. With a=min(T),
F_i=sum_jT_ij(LSE(w)-w_j)>=a sum_j(M-w_j)=a*nM. Averaging and the Gram/h
bound give the stated radius. A stronger-sounding RANGE versus norm inequality
used in Fable012's prose is false for(1,1,-1,-1);it is NOT used here.

Fixed dual points are T;weighted softmax at B;one/two actual-Gram moment
projections. The projection uses solve(Ghat,M),not an assumed identity Gram.
Center residuals per row and shrink toward T to retain >=half the ROW minimum
T. In v1 the code used half each coordinate T_ij,which imposed a stronger
unintended dual constraint and materially changed already-positive stationary
duals. V2 uses floor_i=.5*min_jT_ij and factor=min(1,min_{R_ij<0}
(T_ij-floor_i)/(-R_ij)),where R=Q_trial-T after row centering. All v1wide
intervals remain preserved as valid but inconclusive numerical evidence.
All candidates are savedFP64. For Q>0 the mathematical dual is EXACT
Q*=s_i Q_i/sum_jQ_ij,where s_i=sum_jT_ij in real arithmetic. The numerical
extended-precision q approximation carries a positive relative16gamma(n+4)
envelope,not a claim that floating row sums are exact.

Compute weighted entropy as -sum qlogq+sum slog s. Positive mass-sum error,
q-rescaling error and log Lipschitz bounds are explicit. Coarse arithmetic
allowances use a DOMINANCE ENVELOPE K=32Nn+32N+100 and data magnitudes,not an
actual instruction count or fixed1e-13. K exceeds the relevant rounding count.
Magnitude factors dominate operands and subtraction operands;gamma_K absorbs
compounding. The stableLSE ledger includes score subtraction(<=2maxabsz),exp
relative errors,positive sum,log of a sum>=1,addition of the maximum,mass*LSE,
T*z products/mixed-sign sum and final mean. The primal magnitude is
mean[s(1+4maxabsz+2logn)+sum T|z|]. Entropy uses1+mean(s)+
mean[sum|qlogq|+|slogs|],including the explicit mass term for log absolute error.
Factor4 envelopes supply additional slack. These are conditional error-model
bounds with the operation/operand ledger,not an observed benchmark precision.
V2 moment errors are propagated in whitened coordinates directly:
Z_L=fl_L(XW),E_Z=4gamma_(d+4)|X||W|,
Mhat=fl_L(Z_L.T(qhat-T)/N). With E_R covering q*-qhat and subtraction,
E_M=4gamma_(N+4)|Z_L|.T|Rhat|/N+
4[|Z_L|.T E_R+E_Z.T(|Rhat|+E_R)]/N.
This avoids amplification of an uncentered X moment's roundoff by |W|;
it preserves the same exact M=(XW).T(Q*-T)/N and the factor4 error model.
No error tolerance or learning/optimality threshold is enlarged. Penalize the
entropy lower bound by R*(||M||F+error_F). Include the unconditional weighted
target entropy lower bound,whose moment is exactly zero mathematically.

Arithmetic assumptions:IEEE round-to-nearest,normal finite intermediates,
no FTZ/overflow;dot-product absolute error<=gamma_k sum|products| with
conservative factor4 envelopes. For long-double libm functions assume
|computed_log(x)-log(x)|<=8u*(1+|log(x)|),and exp relative error<=8u on the
stable nonpositive exponent range. These libm bounds are ASSUMPTIONS,not
proved by NumPy. Predetermined high-precision log spots check some values
and cannot prove all arguments. Production requires nmant(longdouble)>52;
Macarm64 fixtures do not substitute for the verified amd64 runtime(nmant63).

Every reported interval is conditional on this error model,not a universal
formal machine-arithmetic certificate. Tight means actual U-L<=1e-6 AND
all rank/error guards passed. Failure/wide bounds mean UNKNOWN optimality.
This is TRAIN-B raw affine only,not validation/population/full-native-class gap.
The final actual raw learners require separate physical-coefficient upper
objectives,all15 actual label-transport checks and the predeclared1e-4 gate.

Native learner objectives use the FULL original Phi matrix plus bias and the
exact stored FP32 native weight/bias arrays. Do not numerically fold constant
columns into a new bias:that introduces a rounding term. Only the canonical
solver problem uses the proved bias-span reduction;the full native point has
exactly the same real function class onB. Deployed CUDA CE remains a separate
machine-function report. All30 raw learner points are retained,both arms/all5.

Post-fit guard failures preserve attained.npz(coefficients,W) and an explicit
ATTAINED_OPTIMALITY_UNKNOWN receipt before the audit begins. A numerical guard
failure retains failure.json and never creates a tight certificate. Production
preflight must exercise amd64 longdouble conversion/libm/subnormal behavior,
nonunit rowmass,tiny positive targets and no-SVD fallback before real fitting.

SciPy's ftol/gtol are stopping tests,not global-gap certificates:
https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-lbfgsb.html

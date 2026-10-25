#!/usr/bin/env python3
"""Standalone mathematical checks; Python standard library only, no data/model.

The rational separation certificate is distinct from Decimal evaluations and
optional floating-point sanity samples. None is a machine-checked formal proof.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
import math
from pathlib import Path
import platform
import random


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dec(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def counterexample():
    p = [F(1, 2), F(2, 5), F(1, 10)]
    q = [F(2, 5), F(2, 5), F(1, 5)]
    gradient = [b-a for a, b in zip(p, q)]
    # At z*=(Lambda,Lambda,-Lambda), these signs certify box KKT.
    require(gradient[0] <= 0 and gradient[1] <= 0 and gradient[2] >= 0,
            'KKT sign failure')
    require(sum(gradient) == 0 and sum(p) == sum(q) == 1, 'Mass failure')
    regret = p[0] - p[1]
    eta = F(1, 1000)
    upper = eta / 2
    lower = regret * regret / 2
    require(lower > upper, 'Exact separation failure')
    require(regret < F(3, 7), 'Printed first-branch condition failed')
    # log(2) >= 1/2 by integrating 1/t >= 1/2 on [1,2].
    require(0 < eta < F(1, 2), 'Strict feasible-score range failed')
    delta = F(1, 10**7)
    transfer = [q[0]+delta, q[1]-delta, q[2]]
    require(all(F(1, 5) <= x <= F(1, 2) for x in transfer), 'Marginal bounds failed')
    require(sum(transfer) == 1 and transfer[0] > 2*transfer[2], 'Joint obstruction failed')
    # E.1 p29: their swap-plus-transfer family, starting from a strict wrong prediction.
    strict_q = [F(39, 100), F(40, 100), F(21, 100)]
    tau1, tau2 = strict_q[1], strict_q[0]
    interval = [max(F(1, 5)-tau2, tau1-F(1, 2)),
                min(F(1, 2)-tau2, tau1-F(1, 5))]
    mu = F(-3, 100)
    transformed = [tau1-mu, tau2+mu, strict_q[2]]
    require(max(strict_q) <= 2*min(strict_q) and strict_q[1] > strict_q[0], 'Strict source infeasible')
    require(interval[0] <= mu <= interval[1], 'mu outside printed interval')
    require(sum(transformed) == 1 and all(F(1, 5) <= x <= F(1, 2) for x in transformed),
            'Transformed marginals invalid')
    require(transformed[0] > 2*transformed[2], 'Printed family obstruction not reproduced')
    with localcontext() as ctx:
        ctx.prec = 80
        lam = Decimal(2).ln()/2
        reference_ce = -sum(dec(a)*dec(b).ln() for a, b in zip(p, q))
        values = []
        for power in range(1, 13):
            e = Decimal(10) ** (-power)
            z = [lam-e, lam, -lam]
            exps = [x.exp() for x in z]; mass = sum(exps)
            probs = [x/mass for x in exps]
            direct = -sum(dec(a)*b.ln() for a, b in zip(p, probs))-reference_ce
            closed = ((2*(-e).exp()+3)/5).ln()+e/2
            require(abs(direct-closed) < Decimal('1e-70'), 'Independent CE evaluations disagree')
            require(direct > 0 and z[1] > max(z[0], z[2]), 'Strict sequence failed')
            values.append(dict(eta=str(e), CE_excess=str(closed), regret=str(dec(regret))))
        t = dec(regret)
        psi = ((1+t)*(1+t).ln()+(1-t)*(1-t).ln())/2
        d = Decimal(values[2]['CE_excess'])
        require(psi > d, 'Decimal counterexample failure')
        numeric = dict(eta='0.001', CE_excess=str(d), printed_required_lower_bound=str(psi),
                       lower_bound_to_excess_ratio=str(psi/d))
    return dict(p=[str(x) for x in p], q_star=[str(x) for x in q],
                gradient_q_minus_p=[str(x) for x in gradient],
                exact_certificate=dict(CE_excess_upper=str(upper), printed_requirement_lower=str(lower),
                    strict_separation=True,
                    justification='D(eta)=log((2 exp(-eta)+3)/5)+eta/2 <= eta/2; '
                    'Psi(0)=Psi_prime(0)=0 and Psi_second(t)=1/(1-t*t)>=1 imply Psi(t)>=t*t/2. '
                    'Convex box KKT certifies z_star; lowering class1 makes class2 uniquely maximal. '
                    'On a singleton both ordinary minimizability gaps are zero.'),
                numeric_80_decimal_digits=numeric, strict_sequence=values,
                marginal_only_transfer=dict(q=[str(x) for x in transfer], delta=str(delta),
                    marginal_bounds_satisfied=True, joint_ratio_constraint_violated=True),
                printed_E1_family_obstruction=dict(source_q=[str(x) for x in strict_q],
                    source_unique_prediction=2, Bayes_label=1, mu=str(mu),
                    printed_mu_interval=[str(x) for x in interval],
                    transformed_q=[str(x) for x in transformed],
                    printed_coordinate_constraints_satisfied=True, transformed_joint_ratio_violated=True,
                    location='NeurIPS 2023 proceedings Appendix E.1 PDF page29, family h_mu and constraint(7)'),
                sharp_residual_n3_R2='1/5',
                distinction='Regret 1/10 disproves a zero residual; it does not alone prove sharp residual 1/5.')


def zheng_variants():
    rows = []
    for p in ([F(17,40), F(2,5), F(7,40)], [F(21,50), F(2,5), F(9,50)]):
        q = [F(2,5), F(2,5), F(1,5)]; eta = F(1,10**6); rho = F(1,4)
        regret = p[0]-p[1]; upper = p[0]*eta
        require(sum(p)==1 and min(p)>0 and max(p)-min(p)<=rho, 'Zheng spread premise failed')
        require(q[0]-p[0]<0 and q[1]-p[1]==0 and q[2]-p[2]>0, 'Zheng box KKT failed')
        require(upper<rho*rho/2 and 2*upper<regret*regret, 'Zheng exact contradiction failed')
        with localcontext() as ctx:
            ctx.prec=80; e=dec(eta)
            excess=((2*(-e).exp()+3)/5).ln()+dec(p[0])*e
            require(excess>0 and (2*excess).sqrt()<dec(regret), 'Zheng Decimal check failed')
            numeric=dict(CE_excess=str(excess), claimed_sqrt_bound=str((2*excess).sqrt()))
        rows.append(dict(p=[str(x) for x in p], posterior_spread=str(max(p)-min(p)),
            allowed_spread=str(rho), eta=str(eta), regret=str(regret), exact_CE_excess_upper=str(upper),
            surrogate_gate=str(rho*rho/2), exact_two_excess_upper=str(2*upper), regret_squared=str(regret*regret),
            decimal_80_digits=numeric))
    return dict(records=rows, class_mapping='H_lin at x=0, B=log(2)/2, any W>0: biases span the full box; '
                'posterior constant off support; both ordinary minimizability gaps are zero.',
                scope='Displayed ICML2023 Theorem3.3 with its ordinary gaps; not an entropy-relative Bayes-calibration bound.')


def sharp_witnesses():
    rows = []
    for n in (3, 4, 10):
        for R in (2, 3, 30):
            m = n-2; Z = 2*R+m; v = F(1, Z); beta = m*v
            t = F(999, 1000)*beta
            p = [R*v+t, R*v]+[v-t/m]*m
            q = [R*v, R*v]+[v]*m
            require(sum(p) == sum(q) == 1 and min(p) > 0, 'Full-support witness mass failure')
            require(p[0]-q[0] == t and p[1] == q[1] and
                    sum(q[i]-p[i] for i in range(2, n)) == t, 'Sharp witness KKT failure')
            slope = (1-beta)*F(R+n-1, R-1)
            require(slope == 1+beta+F(2, R-1), 'Slope expressions disagree')
            require(beta+slope*F(R-1, R+n-1) == 1, 'Slope witness failed')
            rows.append(dict(n=n, R=R, beta=str(beta), full_support_regret=str(t),
                             regret_to_beta='999/1000', slope=str(slope)))
    return rows


def softmax(z):
    exps = [math.exp(x-max(z)) for x in z]; mass = math.fsum(exps)
    return [x/mass for x in exps]


def ce(p, z):
    m = max(z)
    return m+math.log(math.fsum(math.exp(x-m) for x in z))-math.fsum(a*b for a, b in zip(p, z))


def numerical_sanity(samples, seed):
    rng = random.Random(seed); tol = 1e-10
    worst_bridge = worst_finite = -math.inf; min_excess = math.inf; coordinates = 0
    for k in range(samples):
        n = (2, 3, 4, 10, 40)[k % 5]; R = (1.01, 2., 3., 30., 300.)[(k//5) % 5]
        if k % 7 == 0:
            p = [0.]*n; p[rng.randrange(n)] = 1.
        else:
            raw = [rng.gammavariate((.2, 1., 5.)[k % 3], 1.) for _ in range(n)]
            if k % 7 == 1: raw[rng.randrange(n)] = 0.
            mass = math.fsum(raw); p = [x/mass for x in raw]
        lo, hi = 1/(n*R), 1/n
        for _ in range(100):
            v = (lo+hi)/2
            if math.fsum(min(R*v, max(v, x)) for x in p) > 1: hi = v
            else: lo = v
        v = (lo+hi)/2; q = [min(R*v, max(v, x)) for x in p]
        logs = [math.log(x) for x in q]; center = (max(logs)+min(logs))/2
        zs = [x-center for x in logs]; qs = softmax(zs)
        beta = (n-2)/(2*R+n-2); A = 1+beta+2/(R-1)
        i = max(range(n), key=p.__getitem__)
        for j in range(n):
            violation = p[i]-p[j]-beta-A*(qs[i]-qs[j])
            worst_bridge = max(worst_bridge, violation); coordinates += 1
            require(violation <= tol, 'Affine bridge numerical violation')
        lam = math.log(R)/2
        if k % 2:
            amplitude = 10**rng.uniform(-8, 0)
            z = [min(lam, max(-lam, x+rng.uniform(-amplitude, amplitude))) for x in zs]
        else: z = [rng.uniform(-lam, lam) for _ in p]
        excess = ce(p, z)-ce(p, zs); min_excess = min(min_excess, excess)
        require(excess >= -tol, 'Numerical box oracle inconsistency')
        bound = min(1., beta+A*math.sqrt(2*max(0., excess)))
        for j in range(n):
            if z[j] == max(z):
                violation = max(p)-p[j]-bound; worst_finite = max(worst_finite, violation)
                require(violation <= tol, 'Finite-bound numerical violation')
    return dict(samples=samples, seed=seed, bridge_coordinates=coordinates,
                absolute_tolerance=tol, maximum_bridge_violation=worst_bridge if samples else None,
                maximum_finite_bound_violation=worst_finite if samples else None,
                minimum_unclamped_excess=min_excess if samples else None,
                scope='New deterministic floating-point sanity checks with a scalar clipping oracle; '
                      'not exhaustive or formal proof.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); require(args.samples >= 0, 'samples must be nonnegative')
    result = dict(status='PASS', python=platform.python_version(), counterexample=counterexample(),
                  zheng_displayed_bound=zheng_variants(),
                  rational_sharp_witnesses=sharp_witnesses(), numerical_sanity=numerical_sanity(args.samples, args.seed),
                  limits='No external paper parsing, training, inference, private arrays, network access or formal proof assistant. '
                         'The exact separation argument is provided; Decimal and float evaluations are not interval certificates.')
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output: args.output.write_text(text)
    else: print(text, end='')


if __name__ == '__main__': main()

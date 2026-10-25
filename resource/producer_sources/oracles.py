"""Float64 conditional CE oracles with explicit convex first-order gap checks.

Numerical certificates have stated tolerances; they are not outward intervals.
"""
import numpy as np


def logs(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def project_ball(z, radius):
    return z * np.minimum(1., radius / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-300))


def loss(p, z):
    return -(p * logs(z)).sum(axis=-1)


def fw_gap(p, z, domain, budget):
    gradient = np.exp(logs(z)) - p
    support = budget * (np.abs(gradient).sum(axis=-1) if domain == 'box' else np.linalg.norm(gradient, axis=-1))
    return np.maximum(0., (gradient * z).sum(axis=-1) + support)


def box(p, ratio):
    p = np.asarray(p, dtype=np.float64)
    low = np.zeros(p.shape[:-1] + (1,)); high = np.ones_like(low)
    for _ in range(100):
        middle = (low + high) / 2
        below = np.clip(p, middle, ratio * middle).sum(axis=-1, keepdims=True) < 1
        low = np.where(below, middle, low); high = np.where(below, high, middle)
    lower = (low + high) / 2
    q = np.clip(p, lower, ratio * lower)
    z = np.log(q); z -= (z.max(axis=-1, keepdims=True) + z.min(axis=-1, keepdims=True)) / 2
    cap = np.log(ratio) / 2
    assert np.max(abs(z)) <= cap + 1e-10
    q = np.exp(logs(z))
    gap = fw_gap(p, z, 'box', cap)
    assert gap.max(initial=0) <= 1e-10
    return dict(z=z, q=q, CE=loss(p, z), gap=gap, lower_probability=lower)


def ball(p, radius, tolerance=1e-10):
    p = np.asarray(p, dtype=np.float64)
    assert p.ndim == 2
    raw = np.log(np.maximum(p, 1e-300)); raw -= raw.mean(axis=1, keepdims=True)
    interior = np.linalg.norm(raw, axis=1) <= radius
    z = project_ball(raw, radius)
    gradient = np.exp(logs(z)) - p
    multiplier = np.maximum(1e-5, -(gradient * z).sum(1) / max(radius**2, 1e-30))
    active = ~interior

    def inverse(q, lam, v):
        diagonal = q + lam[:, None]
        iq = q / diagonal
        iv = v / diagonal
        denominator = np.maximum(1e-14, 1 - (q * iq).sum(1))
        return iv + iq * ((q * iv).sum(1) / denominator)[:, None]

    for _ in range(60):
        gap = fw_gap(p, z, 'ball', radius)
        ids = np.flatnonzero(active & (gap > tolerance))
        if len(ids) == 0:
            break
        zz, pp, lam = z[ids], p[ids], multiplier[ids]
        q = np.exp(logs(zz)); g = q - pp + lam[:, None] * zz
        constraint = ((zz * zz).sum(1) - radius**2) / 2
        a, b = inverse(q, lam, g), inverse(q, lam, zz)
        dlam = (constraint - (zz * a).sum(1)) / np.maximum((zz * b).sum(1), 1e-30)
        dz = -a - b * dlam[:, None]
        merit = np.linalg.norm(g, axis=1) + abs(constraint)
        step = np.ones(len(ids)); accepted = np.zeros(len(ids), dtype=bool)
        for _ in range(30):
            candidate = zz + step[:, None] * dz
            candidate -= candidate.mean(1, keepdims=True)
            next_lam = lam + step * dlam
            residual = np.exp(logs(candidate)) - pp + next_lam[:, None] * candidate
            next_merit = np.linalg.norm(residual, axis=1) + abs(((candidate**2).sum(1) - radius**2) / 2)
            good = (~accepted) & (next_lam > 0) & (next_merit <= merit * (1 - .0001 * step) + 1e-14)
            z[ids[good]] = candidate[good]; multiplier[ids[good]] = next_lam[good]
            accepted |= good
            if accepted.all():
                break
            step[~accepted] *= .5
        if not accepted.any():
            break
    z = project_ball(z, radius)
    # Independent algorithmic fallback, never silently accept an uncertified oracle.
    for _ in range(20000):
        gap = fw_gap(p, z, 'ball', radius)
        ids = np.flatnonzero(gap > tolerance)
        if len(ids) == 0:
            break
        gradient = np.exp(logs(z[ids])) - p[ids]
        z[ids] = project_ball(z[ids] - 2 * gradient, radius)
    gap = fw_gap(p, z, 'ball', radius)
    assert gap.max(initial=0) <= tolerance * 1.01, gap.max()
    return dict(z=z, q=np.exp(logs(z)), CE=loss(p, z), gap=gap)


def prediction_certificate(p, candidate_scores, oracle):
    """Conservative numerical bounds accounting for approximate oracle optimality."""
    labels = p.argmax(1)
    q = oracle['q']
    alternatives = q.copy(); alternatives[np.arange(len(p)), labels] = -np.inf
    measured_gap = q[np.arange(len(p)), labels] - alternatives.max(1)
    uncertainty = np.sqrt(2 * (oracle['gap'] + 1e-12))
    gap_lower = np.maximum(0., measured_gap - uncertainty)
    ce_excess_upper = np.maximum(0., loss(p, candidate_scores) - oracle['CE'] + oracle['gap'] + 1e-12)
    certified = 2 * ce_excess_upper < gap_lower**2
    return dict(gap_lower=gap_lower, CE_excess_upper=ce_excess_upper, certified=certified)

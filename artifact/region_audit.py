"""
region_audit: reusable image-region constructors and witness-audit signals for
region-tagged Marabou verification.

This module packages the six region constructions (R1-R6) and the three
witness-audit signals used in the TinyTaxiNet/MNIST region-relative
verification study so they can be applied to Marabou-verifiable networks over
flattened image inputs. The application-specific parts (dataset loading,
subset predicates, neighbor definitions, network files) stay in instantiation
scripts; everything here is parameterized by image grid shape and calibration
data.

Region model
------------
A Region is an object with a tag, a params dict, and an attach(net, ivars)
method that adds the region's bounds and constraints to a Marabou network
whose flattened input variables are `ivars`. Math constraints (the R6 family
and the mass invariant) are Constraint objects; `intersect(region, *cs)`
produces a new tagged region.

    R1(S)            r1_subset_box(images)
    R1(x0, eps)      r1_linf_ball(x0, eps)
    R2(.)            intersect(r1, mass_equality(target))     # exact invariant
                     intersect(r1, mass_band(lo, hi))         # statistical band
    R3               r3_inline_relu_decoder(W1,b1,W2,b2,lo,hi)    # in-Marabou
                     r3_latent_box(lo, hi[, mass_output])     # pre-composed net
    R4               r4_local_pca(x0, basis, sigmas, alpha)
    R5               r5_neighbor_box(neighbor_images)
    R6 family        tv_bound(shape, tau), fourier_hf_bound(shape, tau),
                     photometric_bound(b1, b2, eps),
                     per_pair_bound(shape, taus),
                     per_row_mean_band(shape, lo, hi)

Witness audit
-------------
WitnessAudit(reference_images, shape, classifier=None, pair_quantile=99)
calibrates the per-pair Lipschitz thresholds on the reference set and scores
any witness vector with the three signals of the study: plausibility-
classifier probability, nearest-real L2 distance, and per-pair Lipschitz
violation fraction. `baseline(sample)` reports the same signals on real
images so witness scores are always read against a same-domain reference.

Queries
-------
run_query(make_net, region, postcondition, timeout_s) builds a fresh network,
attaches the region, applies the postcondition (the *negated* property, as
usual for Marabou), solves, and returns a verdict dict tagged with the
region's tag and params. Postcondition helpers: output_at_least,
output_at_most, class_confusion.
"""

import time

import numpy as np


# ---------------------------------------------------------------------------
# Grid helpers and calibration
# ---------------------------------------------------------------------------

def grid_pairs(rows, cols):
    """Horizontal/vertical adjacent index pairs on a rows x cols grid."""
    pairs = []
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                pairs.append((i * cols + j, i * cols + j + 1))
            if i + 1 < rows:
                pairs.append((i * cols + j, (i + 1) * cols + j))
    return pairs


def calibrate_per_pair(images, shape, q=99):
    """Per-adjacent-pair |delta| quantile thresholds over a reference set."""
    pairs = grid_pairs(*shape)
    return pairs, np.array([
        float(np.percentile(np.abs(images[:, a] - images[:, b]), q))
        for a, b in pairs
    ])


def local_pca_basis(x0, neighbor_images, d=8):
    """Top-d principal directions and per-direction sigmas of the
    neighbor-deviation matrix around x0. Returns (V_top (d,n_pix), sigmas (d,),
    variance_explained)."""
    D = neighbor_images.astype(np.float64) - x0[None, :]
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    V_top = Vt[:d]
    sigmas = (D @ V_top.T).std(axis=0)
    var_explained = float((S[:d] ** 2).sum() / (S ** 2).sum())
    return V_top, sigmas, var_explained


def load_relu_classifier(npz_path, keys=("c1_W", "c1_b", "c2_W", "c2_b",
                                         "score_W", "score_b"),
                         transpose=False):
    """Load a 2-hidden-layer ReLU + linear-logit classifier from .npz and
    return a callable x -> probability. Set transpose=True when the stored
    kernels are (out, in) rather than (in, out)."""
    w = np.load(npz_path)
    layers = [(w[keys[0]], w[keys[1]]),
              (w[keys[2]], w[keys[3]]),
              (w[keys[4]], w[keys[5]])]
    if transpose:
        layers = [(W.T, b) for W, b in layers]

    def prob(x):
        h = np.atleast_2d(np.asarray(x, dtype=np.float64))
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        logit = np.clip(h.flatten(), -30, 30)
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(p[0]) if p.size == 1 else p

    return prob


# ---------------------------------------------------------------------------
# Marabou encoding primitives
# ---------------------------------------------------------------------------

def _abs_slack(net, var_list, coeffs, offset=0.0):
    """Introduce d >= 0 with d >= |sum_i coeffs_i * var_i + offset| via the
    standard two-inequality slack encoding. Returns the slack variable."""
    dv = int(net.getNewVariable())
    net.setLowerBound(dv, 0.0)
    # q - d <= -offset  and  -q - d <= offset
    net.addInequality(list(var_list) + [dv], list(coeffs) + [-1.0],
                      float(-offset))
    net.addInequality(list(var_list) + [dv], [-c for c in coeffs] + [-1.0],
                      float(offset))
    return dv


class Constraint:
    """A linear-in-pixels constraint attachable to (net, ivars)."""

    def __init__(self, name, params, attach_fn):
        self.name = name
        self.params = params
        self._attach = attach_fn

    def attach(self, net, ivars):
        self._attach(net, ivars)


class Region:
    """A tagged input region attachable to (net, ivars)."""

    def __init__(self, tag, params, attach_fn, constraints=()):
        self.tag = tag
        self.params = dict(params)
        self._attach = attach_fn
        self.constraints = list(constraints)

    def attach(self, net, ivars):
        self._attach(net, ivars)
        for c in self.constraints:
            c.attach(net, ivars)

    def describe(self):
        d = {"tag": self.tag, **self.params}
        if self.constraints:
            d["constraints"] = [{"name": c.name, **c.params}
                                for c in self.constraints]
        return d


def intersect(region, *constraints, tag=None):
    """Region intersected with additional constraints, under a new tag."""
    return Region(tag or region.tag, region.params, region._attach,
                  list(region.constraints) + list(constraints))


# ---------------------------------------------------------------------------
# R1 / R2: pixel boxes and mass invariants
# ---------------------------------------------------------------------------

def r1_subset_box(images, clip=(0.0, 1.0), subset_name=None):
    """R1(S): coordinate-wise [min, max] envelope over an image subset."""
    lo = images.min(axis=0)
    hi = images.max(axis=0)
    if clip is not None:
        lo = np.maximum(lo, clip[0])
        hi = np.minimum(hi, clip[1])

    def attach(net, ivars):
        for i, v in enumerate(ivars):
            net.setLowerBound(int(v), float(lo[i]))
            net.setUpperBound(int(v), float(hi[i]))

    return Region("R1", {"kind": "subset_box", "subset": subset_name,
                         "n_images": int(len(images))}, attach)


def r1_linf_ball(x0, eps, clip=(0.0, 1.0)):
    """R1(x0, eps): per-image L_inf ball, optionally clipped."""
    x0 = np.asarray(x0, dtype=np.float64)

    def attach(net, ivars):
        for i, v in enumerate(ivars):
            lo, hi = x0[i] - eps, x0[i] + eps
            if clip is not None:
                lo, hi = max(clip[0], lo), min(clip[1], hi)
            net.setLowerBound(int(v), float(lo))
            net.setUpperBound(int(v), float(hi))

    return Region("R1", {"kind": "linf_ball", "eps": float(eps)}, attach)


def mass_equality(target):
    """Exact preprocessing invariant sum_i x_i = target (TaxiNet mean=0.5)."""
    def attach(net, ivars):
        net.addEquality([int(v) for v in ivars], [1.0] * len(ivars),
                        float(target))
    return Constraint("mass_equality", {"target": float(target)}, attach)


def mass_band(lo, hi):
    """Statistical total-mass band lo <= sum_i x_i <= hi (MNIST R2)."""
    def attach(net, ivars):
        iv = [int(v) for v in ivars]
        net.addInequality(iv, [1.0] * len(iv), float(hi))
        net.addInequality(iv, [-1.0] * len(iv), float(-lo))
    return Constraint("mass_band", {"lo": float(lo), "hi": float(hi)}, attach)


# ---------------------------------------------------------------------------
# R3: autoencoder decoder-box
# ---------------------------------------------------------------------------

def r3_inline_relu_decoder(W1, b1, W2, b2, latent_lo, latent_hi,
                           clip=(0.0, 1.0)):
    """R3 for a 2-layer decoder latent(d) -> ReLU(h) -> linear(n_pix),
    composed in-Marabou: the network's pixel inputs are constrained to equal
    the decoder's output over the latent box."""
    W1 = np.asarray(W1, dtype=np.float64)   # (h, d)
    W2 = np.asarray(W2, dtype=np.float64)   # (n_pix, h)

    def attach(net, ivars):
        d = W1.shape[1]
        h = W1.shape[0]
        z = []
        for j in range(d):
            v = int(net.getNewVariable())
            net.setLowerBound(v, float(latent_lo[j]))
            net.setUpperBound(v, float(latent_hi[j]))
            z.append(v)
        post = []
        for k in range(h):
            pre = int(net.getNewVariable())
            pst = int(net.getNewVariable())
            net.setLowerBound(pst, 0.0)
            net.addEquality(z + [pre], list(W1[k]) + [-1.0], float(-b1[k]))
            net.addRelu(pre, pst)
            post.append(pst)
        for i, v in enumerate(ivars):
            net.addEquality(post + [int(v)], list(W2[i]) + [-1.0],
                            float(-b2[i]))
            if clip is not None:
                net.setLowerBound(int(v), float(clip[0]))
                net.setUpperBound(int(v), float(clip[1]))

    return Region("R3", {"kind": "inline_relu_decoder",
                         "latent_dim": int(W1.shape[1])}, attach)


def r3_latent_box(latent_lo, latent_hi, mass_output=None):
    """R3 for a pre-composed decoder-circ-network (e.g., ONNX): the network's
    inputs ARE the latents; bound them to the training latent box. If the
    composition exposes an auxiliary decoded-pixel-sum output, pass
    mass_output=(output_var_index, target) to enforce the mass invariant."""
    def attach(net, ivars):
        for j, v in enumerate(ivars):
            net.setLowerBound(int(v), float(latent_lo[j]))
            net.setUpperBound(int(v), float(latent_hi[j]))
        if mass_output is not None:
            ovar, target = mass_output
            net.setLowerBound(int(ovar), float(target))
            net.setUpperBound(int(ovar), float(target))

    return Region("R3", {"kind": "latent_box",
                         "latent_dim": int(len(latent_lo))}, attach)


# ---------------------------------------------------------------------------
# R4 / R5: local PCA and neighbor boxes
# ---------------------------------------------------------------------------

def r4_local_pca(x0, V_top, sigmas, alpha, clip=(0.0, 1.0)):
    """R4: x = x0 + V_top^T z with |z_j| <= alpha * sigma_j, x clipped."""
    x0 = np.asarray(x0, dtype=np.float64)
    V_top = np.asarray(V_top, dtype=np.float64)   # (d, n_pix)

    def attach(net, ivars):
        d = V_top.shape[0]
        z = []
        for j in range(d):
            v = int(net.getNewVariable())
            bound = float(alpha * sigmas[j])
            net.setLowerBound(v, -bound)
            net.setUpperBound(v, bound)
            z.append(v)
        for i, v in enumerate(ivars):
            net.addEquality(z + [int(v)], list(V_top[:, i]) + [-1.0],
                            float(-x0[i]))
            if clip is not None:
                net.setLowerBound(int(v), float(clip[0]))
                net.setUpperBound(int(v), float(clip[1]))

    return Region("R4", {"kind": "local_pca", "d": int(V_top.shape[0]),
                         "alpha": float(alpha)}, attach)


def r5_neighbor_box(neighbor_images, clip=(0.0, 1.0)):
    """R5: per-pixel [min, max] over a label- or class-conditioned neighbor
    set (the neighbor selection rule lives in the instantiation)."""
    return Region("R5", {"kind": "neighbor_box",
                         "n_neighbors": int(len(neighbor_images))},
                  r1_subset_box(neighbor_images, clip)._attach)


# ---------------------------------------------------------------------------
# R6 family: linear-in-pixel math filters
# ---------------------------------------------------------------------------

def tv_bound(shape, tau):
    """Total variation sum over adjacent pairs of |x_a - x_b| <= tau."""
    pairs = grid_pairs(*shape)

    def attach(net, ivars):
        slacks = [_abs_slack(net, [int(ivars[a]), int(ivars[b])], [1.0, -1.0])
                  for a, b in pairs]
        net.addInequality(slacks, [1.0] * len(slacks), float(tau))

    return Constraint("tv", {"tau": float(tau)}, attach)


def per_pair_bound(shape, taus):
    """Per-adjacent-pair Lipschitz bound |x_a - x_b| <= tau_ab."""
    pairs = grid_pairs(*shape)

    def attach(net, ivars):
        for (a, b), t in zip(pairs, taus):
            net.addInequality([int(ivars[a]), int(ivars[b])], [1.0, -1.0],
                              float(t))
            net.addInequality([int(ivars[a]), int(ivars[b])], [-1.0, 1.0],
                              float(t))

    return Constraint("per_pair", {"tau_median": float(np.median(taus))},
                      attach)


def fourier_hf_bound(shape, tau, orthonormal=True):
    """High-frequency Fourier L1 bound: sum over HF coefficients of
    |Re c| + |Im c| <= tau, with HF = {(k,l): k >= rows/2 or l >= cols/2}.
    Linear in pixels; the orthonormal flag selects the 1/sqrt(n_pix) DFT
    convention used in the paper."""
    rows, cols = shape
    n_pix = rows * cols
    norm = 1.0 / np.sqrt(n_pix) if orthonormal else 1.0
    terms = []
    r_idx = np.arange(rows)
    s_idx = np.arange(cols)
    for k in range(rows):
        for l in range(cols):
            if not (k >= rows // 2 or l >= cols // 2):
                continue
            phase = -2.0 * np.pi * (k * r_idx[:, None] / rows
                                    + l * s_idx[None, :] / cols)
            terms.append(norm * np.cos(phase).flatten())
            terms.append(norm * np.sin(phase).flatten())

    def attach(net, ivars):
        iv = [int(v) for v in ivars]
        slacks = [_abs_slack(net, iv, list(coefs)) for coefs in terms]
        net.addInequality(slacks, [1.0] * len(slacks), float(tau))

    return Constraint("fourier_hf",
                      {"tau": float(tau), "orthonormal": bool(orthonormal)},
                      attach)


def photometric_bound(b1, b2, eps):
    """||x - alpha*b1 - beta*b2||_inf <= eps with free (alpha, beta)."""
    b1 = np.asarray(b1, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    def attach(net, ivars):
        alpha = int(net.getNewVariable())
        beta = int(net.getNewVariable())
        net.setLowerBound(alpha, -1e3); net.setUpperBound(alpha, 1e3)
        net.setLowerBound(beta, -1e3); net.setUpperBound(beta, 1e3)
        for i, v in enumerate(ivars):
            net.addInequality([int(v), alpha, beta],
                              [1.0, -float(b1[i]), -float(b2[i])], float(eps))
            net.addInequality([int(v), alpha, beta],
                              [-1.0, float(b1[i]), float(b2[i])], float(eps))

    return Constraint("photometric", {"eps": float(eps)}, attach)


def per_row_mean_band(shape, lo_per_row, hi_per_row):
    """Per-row mean band: lo_r <= mean(row r) <= hi_r."""
    rows, cols = shape

    def attach(net, ivars):
        for r in range(rows):
            iv = [int(ivars[r * cols + s]) for s in range(cols)]
            net.addInequality(iv, [1.0 / cols] * cols, float(hi_per_row[r]))
            net.addInequality(iv, [-1.0 / cols] * cols, float(-lo_per_row[r]))

    return Constraint("per_row_mean", {}, attach)


# ---------------------------------------------------------------------------
# Postconditions (negated properties) and the query runner
# ---------------------------------------------------------------------------

def output_at_least(out_index, bound):
    """Negated upper-bound property: exists x with y[out_index] >= bound."""
    def apply(net, ovars):
        net.setLowerBound(int(ovars[out_index]), float(bound))
    apply.params = {"post": f"y[{out_index}] >= {bound}"}
    return apply


def output_at_most(out_index, bound):
    """Negated lower-bound property: exists x with y[out_index] <= bound."""
    def apply(net, ovars):
        net.setUpperBound(int(ovars[out_index]), float(bound))
    apply.params = {"post": f"y[{out_index}] <= {bound}"}
    return apply


def class_confusion(true_index, target_index):
    """Negated margin property: exists x with y[target] >= y[true]."""
    def apply(net, ovars):
        net.addInequality([int(ovars[true_index]), int(ovars[target_index])],
                          [1.0, -1.0], 0.0)
    apply.params = {"post": f"y[{target_index}] >= y[{true_index}]"}
    return apply


def run_query(make_net, region, postcondition, timeout_s=300,
              marabou_options=None):
    """Build a fresh network via make_net() -> (net, ivars, ovars), attach
    the region and the (negated) postcondition, solve, and return a tagged
    verdict dict. SAT verdicts include the witness input vector and outputs."""
    from maraboupy import Marabou
    net, ivars, ovars = make_net()
    region.attach(net, ivars)
    postcondition(net, ovars)
    opts = marabou_options or Marabou.createOptions(
        verbosity=0, timeoutInSeconds=timeout_s)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    out = {
        "status": str(status).strip().lower(),
        "time_seconds": round(time.time() - t0, 4),
        "region": region.describe(),
        **getattr(postcondition, "params", {}),
    }
    if out["status"] == "sat" and vals:
        out["witness"] = [float(vals[int(v)]) for v in ivars]
        out["outputs"] = [float(vals[int(v)]) for v in ovars]
    return out


# ---------------------------------------------------------------------------
# Witness audit
# ---------------------------------------------------------------------------

class WitnessAudit:
    """Three-signal witness audit calibrated on a reference image set:
    plausibility-classifier probability (optional, learned), nearest-real L2
    (not learned), and per-pair Lipschitz violation fraction (not learned)."""

    def __init__(self, reference_images, shape, classifier=None,
                 pair_quantile=99):
        self.reference = np.asarray(reference_images, dtype=np.float64)
        self.shape = tuple(shape)
        self.classifier = classifier
        self.pairs, self.taus = calibrate_per_pair(self.reference, shape,
                                                   q=pair_quantile)

    def _per_pair_frac(self, x):
        viol = sum(abs(x[a] - x[b]) > t
                   for (a, b), t in zip(self.pairs, self.taus))
        return viol / len(self.pairs)

    def score(self, x):
        x = np.asarray(x, dtype=np.float64)
        rec = {
            "nearest_real_L2": float(np.min(
                np.linalg.norm(self.reference - x[None, :], axis=1))),
            "per_pair_violation_frac": float(self._per_pair_frac(x)),
        }
        if self.classifier is not None:
            rec["classifier_prob"] = float(self.classifier(x))
        return rec

    def baseline(self, sample=None, exclude_self=True):
        """Score real images so witness scores have a same-domain reference.
        With exclude_self, the L2 signal is the distance to the nearest
        OTHER reference image."""
        X = self.reference if sample is None else np.asarray(sample)
        l2, ppf, probs = [], [], []
        for x in X:
            d = np.linalg.norm(self.reference - x[None, :], axis=1)
            if exclude_self:
                d = d[d > 1e-12]
            l2.append(float(d.min()))
            ppf.append(self._per_pair_frac(x))
            if self.classifier is not None:
                probs.append(float(self.classifier(x)))
        out = {
            "n": int(len(X)),
            "nearest_real_L2_median": float(np.median(l2)),
            "per_pair_violation_frac_mean": float(np.mean(ppf)),
        }
        if probs:
            out["classifier_prob_mean"] = float(np.mean(probs))
            out["classifier_prob_median"] = float(np.median(probs))
        return out

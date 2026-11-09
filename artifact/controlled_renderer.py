#!/usr/bin/env python3
"""Part B: a tiny controlled lane/runway renderer with known semantic truth.

Every image is produced from explicit state parameters, so the ground-truth
cross-track error (CTE) of an image is known by construction, and we can later
ask whether a solver witness corresponds to ANY renderable state (semantic
reachability) -- the check that is impossible for TinyTaxiNet because no
camera/observation model is available.

Image: 8 rows x 16 cols grayscale, pixel values in [0,1], flattened to 128.
A bright near-vertical "centerline" stripe sits at column

    c(row) = COL_CENTER + CTE + HEAD_GAIN * heading * (row - 3.5)

so the lateral offset of the stripe encodes CTE (sign included) and heading
tilts it. Brightness, Gaussian noise, and line width are nuisance parameters.

Pure numpy; no ML framework, no Marabou.
"""

from __future__ import annotations

import numpy as np

ROWS, COLS = 8, 16
N_PIX = ROWS * COLS
COL_CENTER = 7.5
HEAD_GAIN = 0.15
LINE_AMP = 0.45

# State ranges for the controlled world.
CTE_RANGE = (-6.0, 6.0)
HEAD_RANGE = (-10.0, 10.0)
BRIGHT_RANGE = (0.30, 0.55)
WIDTH_RANGE = (1.0, 2.0)


def render(cte, heading=0.0, brightness=0.42, width=1.4, noise=0.0, rng=None):
    """Render one (cte, heading, brightness, width) state to a 128-vector."""
    rows = np.arange(ROWS)[:, None]
    cols = np.arange(COLS)[None, :]
    center = COL_CENTER + cte + HEAD_GAIN * heading * (rows - (ROWS - 1) / 2.0)
    img = brightness + LINE_AMP * np.exp(-((cols - center) ** 2) / (2 * width * width))
    if noise and rng is not None:
        img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0.0, 1.0).reshape(-1)


def sample_states(n, rng, cte_range=CTE_RANGE):
    """Sample n random states; return (X images, states dict-of-arrays)."""
    cte = rng.uniform(*cte_range, n)
    heading = rng.uniform(*HEAD_RANGE, n)
    bright = rng.uniform(*BRIGHT_RANGE, n)
    width = rng.uniform(*WIDTH_RANGE, n)
    noise = rng.uniform(0.0, 0.02, n)
    X = np.stack([render(cte[i], heading[i], bright[i], width[i], noise[i], rng)
                  for i in range(n)])
    states = {"cte": cte, "heading": heading, "brightness": bright,
              "width": width, "noise": noise}
    return X, states


def dense_grid(n_grid, rng, cte_range=CTE_RANGE):
    """A dense noise-free renderer grid used as an external semantic-support
    oracle independent of any learned classifier. Returns (G images, states)."""
    cte = rng.uniform(*cte_range, n_grid)
    heading = rng.uniform(*HEAD_RANGE, n_grid)
    bright = rng.uniform(*BRIGHT_RANGE, n_grid)
    width = rng.uniform(*WIDTH_RANGE, n_grid)
    G = np.stack([render(cte[i], heading[i], bright[i], width[i])
                  for i in range(n_grid)])
    states = {"cte": cte, "heading": heading, "brightness": bright,
              "width": width}
    return G, states


def nearest_in_grid(x, G, states):
    """Nearest renderer-grid image to x: (L2, state dict)."""
    d = np.linalg.norm(G - x[None, :], axis=1)
    j = int(np.argmin(d))
    return float(d[j]), {k: float(v[j]) for k, v in states.items()}


def renderable_distance(x, rng, n_grid=4000, cte_range=CTE_RANGE):
    """Convenience: build a one-off grid and return nearest (L2, state)."""
    G, states = dense_grid(n_grid, rng, cte_range)
    return nearest_in_grid(x, G, states)


def rendered_nn_percentile(Xheld, p=99.0):
    """Held-out rendered-to-rendered nearest-neighbour L2 at percentile p.
    Used to define the 'renderer-supported' threshold before seeing witnesses."""
    n = len(Xheld)
    nn = np.empty(n)
    for i in range(n):
        d = np.linalg.norm(Xheld - Xheld[i][None, :], axis=1)
        d[i] = np.inf
        nn[i] = d.min()
    return float(np.percentile(nn, p)), nn


def checkerboard_trigger(pmin, pmax):
    """A non-renderable high-frequency trigger living at a CORNER of the
    per-pixel box: pixel i takes pmax[i] on the 'white' squares of an 8x16
    checkerboard and pmin[i] on the 'black' squares. Reachable inside a broad
    per-pixel box (it is a box corner) but never produced by the renderer."""
    rows = np.arange(ROWS)[:, None]
    cols = np.arange(COLS)[None, :]
    mask = ((rows + cols) % 2 == 0).reshape(-1)
    return np.where(mask, pmax, pmin)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X, st = sample_states(5, rng)
    print("rendered", X.shape, "cte sample", np.round(st["cte"], 2))
    d, s = renderable_distance(X[0], rng)
    print("self renderable distance ~", round(d, 3), "at", {k: round(v, 2) for k, v in s.items()})

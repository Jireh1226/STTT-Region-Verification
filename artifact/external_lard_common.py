#!/usr/bin/env python3.11
"""Shared helpers for the external LARD-V2 region-tagged verification experiment."""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_dataset():
    return np.load(os.path.join(ROOT, "external_lard_dataset.npz"), allow_pickle=True)


def load_target():
    return json.load(open(os.path.join(ROOT, "external_lard_target.json")))


def labels_split(d, split, tgt=None):
    """3-class tercile labels for the chosen target on a given split ('tr' or 'te')."""
    if tgt is None:
        tgt = load_target()
    v = np.asarray(d[tgt["chosen_target"] + "_" + split], dtype=np.float64)
    q1, q2 = tgt["thresholds_train"]
    y = np.ones(len(v), dtype=np.int64)
    y[v <= q1] = 0
    y[v >= q2] = 2
    return y


def load_model():
    w = np.load(os.path.join(ROOT, "external_lard_mlp_weights.npz"), allow_pickle=True)
    Ws = [w["W1"].astype(np.float64), w["W2"].astype(np.float64), w["W3"].astype(np.float64)]
    bs = [w["b1"].astype(np.float64), w["b2"].astype(np.float64), w["b3"].astype(np.float64)]
    return Ws, bs, float(w["q1"]), float(w["q2"]), int(w["H"]), int(w["W"])


def forward(X, Ws, bs):
    a = np.atleast_2d(X).astype(np.float64)
    for i in range(len(Ws) - 1):
        a = np.maximum(a @ Ws[i] + bs[i], 0.0)
    return a @ Ws[-1] + bs[-1]


def labels_from_offset(off, q1, q2):
    y = np.ones(len(np.atleast_1d(off)), dtype=np.int64)
    off = np.atleast_1d(off)
    y[off <= q1] = 0
    y[off >= q2] = 2
    return y


def write_nnet(path, Ws, bs, din, dout, comment="// composed network", in_lo=0.0, in_hi=1.0):
    """NNet writer; raw inputs (means=0, ranges=1 identity normalization).

    in_lo/in_hi set the network's declared input min/max. Use [0,1] for pixel inputs and a
    wide range for composed-PCA latent inputs (which can be negative); the actual region box is
    imposed by the verifier on top of these.
    """
    layer_sizes = [din] + [W.shape[1] for W in Ws]
    nlayers = len(Ws)
    maxw = max(layer_sizes)
    lines = [comment, f"{nlayers},{din},{dout},{maxw},",
             ",".join(str(s) for s in layer_sizes) + ",", "0,",
             ",".join([f"{in_lo}"] * din) + ",",
             ",".join([f"{in_hi}"] * din) + ",",
             ",".join(["0.0"] * (din + 1)) + ",",
             ",".join(["1.0"] * (din + 1)) + ","]
    for li in range(nlayers):
        Wt = np.asarray(Ws[li]).T
        for j in range(Wt.shape[0]):
            lines.append(",".join(f"{v:.8e}" for v in Wt[j]) + ",")
        for j in range(len(bs[li])):
            lines.append(f"{float(bs[li][j]):.8e},")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def compose_pca_nnet(path, Ws, bs, V, c, din_z, dout):
    """Fold affine x = c + V^T z into the first MLP layer; write composed nnet z->logits.

    V: (d, D) principal directions (rows). c: (D,) anchor. z: (d,).
    h1pre[j] = (c @ W1)[j] + b1[j] + sum_k z_k (V @ W1)[k,j]
    => W1' = V @ W1   (d, hidden1) ; b1' = c @ W1 + b1
    """
    W1, b1 = Ws[0], bs[0]
    W1p = V @ W1                      # (d, hidden1)
    b1p = c @ W1 + b1                 # (hidden1,)
    Ws2 = [W1p] + Ws[1:]
    bs2 = [b1p] + bs[1:]
    write_nnet(path, Ws2, bs2, din_z, dout, comment="// R4 local-PCA composed (z -> logits)",
               in_lo=-1e6, in_hi=1e6)
    return Ws2, bs2


def parse_marabou(out):
    """Return (status, witness_dict). witness keys are input-var indices."""
    toks = out.lower().split()
    status = ("unsat" if "unsat" in toks else
              "sat" if "sat" in toks else
              "timeout" if "timeout" in out.lower() else "error")
    wit = {}
    if status == "sat":
        for ln in out.splitlines():
            s = ln.strip()
            # Marabou prints input assignment as `x<idx> = value` (no underscore)
            if (s.startswith("x") and "=" in s and "_" not in s):
                try:
                    k, v = s.split("=")
                    wit[int(k.strip()[1:])] = float(v)
                except ValueError:
                    pass
    return status, wit

#!/usr/bin/env python3.11
"""
Train a COMPACT verifier-tractable ReLU MLP on the external LARD-V2 runway target.

Architecture: 1152 (24x48 grayscale, [0,1]) -> 64 -> 32 -> 3 logits (ReLU hidden, linear out)
Target:       low/mid/high camera pitch terciles selected by select_external_lard_target.py.
Trained in pure NumPy (no torch/GPU). Exports:
  - external_lard_mlp_weights.npz   (W1,b1,W2,b2,W3,b3 + label thresholds)
  - external_lard_mlp.nnet          (Marabou-compatible, raw [0,1] pixel inputs, no normalization)
  - external_lard_training_report.json

Inputs are fed RAW in [0,1] (no standardization) so that pixel-box regions stay simple.
"""
import json, os, time
import numpy as np
from external_lard_common import ROOT, load_dataset, load_target, labels_split

SEED = 0
EPOCHS = 400
BATCH = 128
LR = 2e-3
L2 = 1e-5
HID = [64, 32]


def init_mlp(din, hid, dout, rng):
    sizes = [din] + hid + [dout]
    Ws, bs = [], []
    for i in range(len(sizes) - 1):
        fan_in = sizes[i]
        Ws.append((rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / fan_in)).astype(np.float64))
        bs.append(np.zeros(sizes[i + 1], dtype=np.float64))
    return Ws, bs


def forward(X, Ws, bs):
    a = X
    caches = []
    for i in range(len(Ws) - 1):
        z = a @ Ws[i] + bs[i]
        caches.append((a, z))
        a = np.maximum(z, 0.0)
    z = a @ Ws[-1] + bs[-1]
    caches.append((a, z))
    return z, caches


def softmax_ce(logits, y):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    g = p.copy(); g[np.arange(n), y] -= 1.0; g /= n
    return loss, g


def backward(g, caches, Ws):
    grads_W = [None] * len(Ws); grads_b = [None] * len(Ws)
    a_last, _ = caches[-1]
    grads_W[-1] = a_last.T @ g; grads_b[-1] = g.sum(0)
    da = g @ Ws[-1].T
    for i in range(len(Ws) - 2, -1, -1):
        a_in, z = caches[i]
        dz = da * (z > 0)
        grads_W[i] = a_in.T @ dz; grads_b[i] = dz.sum(0)
        da = dz @ Ws[i].T
    return grads_W, grads_b


def accuracy(X, y, Ws, bs):
    logits, _ = forward(X, Ws, bs)
    return float((logits.argmax(1) == y).mean())


def per_class(X, y, Ws, bs, k=3):
    logits, _ = forward(X, Ws, bs)
    pred = logits.argmax(1)
    out = {}
    for c in range(k):
        m = y == c
        out[c] = [int(m.sum()), float((pred[m] == c).mean()) if m.sum() else 0.0]
    return out


def write_nnet(path, Ws, bs, din, dout):
    """NNet format with raw [0,1] inputs (means=0, ranges=1 -> identity normalization)."""
    layer_sizes = [din] + [W.shape[1] for W in Ws]
    nlayers = len(Ws)
    maxw = max(layer_sizes)
    lines = ["// External LARD-V2 runway pitch-attitude classifier; trained by train_external_lard_mlp.py"]
    lines.append(f"{nlayers},{din},{dout},{maxw},")
    lines.append(",".join(str(s) for s in layer_sizes) + ",")
    lines.append("0,")
    lines.append(",".join(["0.0"] * din) + ",")                 # input mins (override by vnnlib)
    lines.append(",".join(["1.0"] * din) + ",")                 # input maxs
    lines.append(",".join(["0.0"] * (din + 1)) + ",")           # means (inputs + output)
    lines.append(",".join(["1.0"] * (din + 1)) + ",")           # ranges (inputs + output)
    for li in range(nlayers):
        Wt = Ws[li].T                                            # [out, in]
        for j in range(Wt.shape[0]):
            lines.append(",".join(f"{v:.8e}" for v in Wt[j]) + ",")
        for j in range(len(bs[li])):
            lines.append(f"{bs[li][j]:.8e},")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rng = np.random.default_rng(SEED)
    d = load_dataset()
    tgt = load_target()
    q1, q2 = tgt["thresholds_train"]
    Xtr_all, Xte = d["Xtr"], d["Xte"]
    ytr_all = labels_split(d, "tr", tgt)
    yte = labels_split(d, "te", tgt)

    # carve a validation split from train (test stays fully held out for anchors)
    perm = rng.permutation(len(Xtr_all))
    nval = max(1, int(0.15 * len(perm)))
    vi, ti = perm[:nval], perm[nval:]
    Xtr, ytr = Xtr_all[ti], ytr_all[ti]
    Xval, yval = Xtr_all[vi], ytr_all[vi]

    din, dout = Xtr.shape[1], 3
    Ws, bs = init_mlp(din, HID, dout, rng)
    mW = [np.zeros_like(W) for W in Ws]; vW = [np.zeros_like(W) for W in Ws]
    mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    t = 0
    n = len(Xtr)
    best = (-1.0, None)
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            logits, caches = forward(Xtr[idx], Ws, bs)
            _, g = softmax_ce(logits, ytr[idx])
            gW, gb = backward(g, caches, Ws)
            t += 1
            for i in range(len(Ws)):
                gW[i] += L2 * Ws[i]
                mW[i] = b1a * mW[i] + (1 - b1a) * gW[i]; vW[i] = b2a * vW[i] + (1 - b2a) * gW[i] ** 2
                mb[i] = b1a * mb[i] + (1 - b1a) * gb[i]; vb[i] = b2a * vb[i] + (1 - b2a) * gb[i] ** 2
                mWh = mW[i] / (1 - b1a ** t); vWh = vW[i] / (1 - b2a ** t)
                mbh = mb[i] / (1 - b1a ** t); vbh = vb[i] / (1 - b2a ** t)
                Ws[i] -= LR * mWh / (np.sqrt(vWh) + eps)
                bs[i] -= LR * mbh / (np.sqrt(vbh) + eps)
        if (ep + 1) % 25 == 0 or ep == 0:
            va = accuracy(Xval, yval, Ws, bs)
            print(f"ep {ep+1:3d}  val_acc {va:.3f}  train_acc {accuracy(Xtr,ytr,Ws,bs):.3f}", flush=True)
            if va > best[0]:
                best = (va, ([W.copy() for W in Ws], [b.copy() for b in bs]))
    if best[1] is not None and best[0] >= accuracy(Xval, yval, Ws, bs):
        Ws, bs = best[1]

    tr_acc = accuracy(Xtr, ytr, Ws, bs)
    val_acc = accuracy(Xval, yval, Ws, bs)
    te_acc = accuracy(Xte, yte, Ws, bs)

    print(f"\nFINAL  train {tr_acc:.3f}  val {val_acc:.3f}  test {te_acc:.3f}", flush=True)
    print("test per-class [n, acc]:", per_class(Xte, yte, Ws, bs), flush=True)

    Ws32 = [W.astype(np.float32) for W in Ws]; bs32 = [b.astype(np.float32) for b in bs]
    np.savez(os.path.join(ROOT, "external_lard_mlp_weights.npz"),
             W1=Ws32[0], b1=bs32[0], W2=Ws32[1], b2=bs32[1], W3=Ws32[2], b3=bs32[2],
             q1=q1, q2=q2, H=int(d["H"]), W=int(d["W"]),
             target=tgt["chosen_target"], class_names=np.array(tgt["class_names"]))
    write_nnet(os.path.join(ROOT, "external_lard_mlp.nnet"), Ws32, bs32, din, dout)

    report = {
        "script": "train_external_lard_mlp.py",
        "arch": f"{din}-{'-'.join(map(str,HID))}-{dout} ReLU MLP, linear logits",
        "target": tgt["chosen_target"], "class_names": tgt["class_names"],
        "input": f"{int(d['H'])}x{int(d['W'])} grayscale, raw [0,1], no standardization",
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "l2": L2,
        "n_train": int(len(Xtr)), "n_val": int(len(Xval)), "n_test": int(len(Xte)),
        "tercile_thresholds_train": [q1, q2],
        "train_acc": tr_acc, "val_acc": val_acc, "test_acc": te_acc,
        "majority_baseline": 1/3,
        "test_per_class": {str(k): v for k, v in per_class(Xte, yte, Ws, bs).items()},
        "gpu_used": False,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(ROOT, "external_lard_training_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("WROTE external_lard_mlp.nnet, external_lard_mlp_weights.npz, external_lard_training_report.json", flush=True)


if __name__ == "__main__":
    main()

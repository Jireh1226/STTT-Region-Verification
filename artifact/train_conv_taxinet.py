#!/usr/bin/env python3
"""Train a small CNN on the TinyTaxiNet task (predict CTE, HE from the 8x16
downsampled image) and export to ONNX, for a conv-architecture scalability probe
of the region-tagged Marabou workflow. Small on purpose (few channels) so Marabou
can attempt it; the point is to measure the conv verification boundary, not to
beat the MLP.
"""
import json
import numpy as np
import h5py
import torch
import torch.nn as nn

DATA = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
ONNX_OUT = "conv_taxinet.onnx"
H, W = 8, 16


class ConvNet(nn.Module):
    def __init__(self, ch=4):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.fc1 = nn.Linear(ch * H * W, 32)
        self.fc2 = nn.Linear(32, 2)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.c1(x))
        x = self.r(self.c2(x))
        x = x.flatten(1)
        x = self.r(self.fc1(x))
        return self.fc2(x)


def main():
    torch.manual_seed(0)
    with h5py.File(DATA, "r") as f:
        X = f["y_train"][:].reshape(-1, 1, H, W).astype(np.float32)
        Y = f["X_train"][:][:, :2].astype(np.float32)     # (CTE, HE)
    n = len(X); idx = np.arange(n); rng = np.random.default_rng(0); rng.shuffle(idx)
    ntr = int(0.9 * n)
    Xtr, Ytr = X[idx[:ntr]], Y[idx[:ntr]]
    Xte, Yte = X[idx[ntr:]], Y[idx[ntr:]]
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xte_t, Yte_t = torch.tensor(Xte), torch.tensor(Yte)

    model = ConvNet(ch=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(40):
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            out = model(Xtr_t[b])
            loss = loss_fn(out, Ytr_t[b])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_mse = float(loss_fn(model(Xtr_t), Ytr_t))
        te_mse = float(loss_fn(model(Xte_t), Yte_t))
        te_pred = model(Xte_t).numpy()
        cte_mae = float(np.mean(np.abs(te_pred[:, 0] - Yte[:, 0])))
        he_mae = float(np.mean(np.abs(te_pred[:, 1] - Yte[:, 1])))
    print(f"train MSE {tr_mse:.4f}  test MSE {te_mse:.4f}  test CTE MAE {cte_mae:.3f}m  HE MAE {he_mae:.3f}deg")

    dummy = torch.zeros(1, 1, H, W)
    torch.onnx.export(model, dummy, ONNX_OUT, opset_version=11,
                      input_names=["image"], output_names=["cte_he"])
    json.dump({"arch": "Conv(1,4,3)-ReLU-Conv(4,4,3)-ReLU-Flatten-Linear(512,32)-ReLU-Linear(32,2)",
               "train_mse": round(tr_mse, 4), "test_mse": round(te_mse, 4),
               "test_cte_mae_m": round(cte_mae, 3), "test_he_mae_deg": round(he_mae, 3),
               "onnx": ONNX_OUT, "input_shape": [1, 1, H, W]},
              open("conv_taxinet_train_report.json", "w"), indent=2)
    print("exported", ONNX_OUT)


if __name__ == "__main__":
    main()

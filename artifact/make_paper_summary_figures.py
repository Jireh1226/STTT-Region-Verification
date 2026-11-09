#!/usr/bin/env python3
"""Generate contribution-level figures for the STTT manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


OUT = Path("paper_submission_sttt")
EDGE = "#222222"
GRID = "#D9D9D9"


plt.rcParams.update(
    {
        "axes.edgecolor": EDGE,
        "axes.linewidth": 0.8,
        "font.size": 8,
        "hatch.linewidth": 0.5,
    }
)


def readable_text_color(color: tuple[float, float, float, float]) -> str:
    r, g, b = color[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.58 else "white"


def make_region_transition_heatmap() -> None:
    rows = [
        "TinyTaxiNet P1\nCTE bound",
        "TinyTaxiNet P3\nrobustness",
        "TinyTaxiNet P4\nwrong sign",
        "MNIST P3\nconfusion",
        "VNN-COMP\nmnist_fc",
        "LARD V2\npitch",
    ]
    cols = [
        "$R_1$\nbroad",
        "$R_2$\ninvariant",
        "$R_3$\nAE",
        "$R_4$\nlocal PCA",
        "Rand.\n8D",
        "$R_5$\nneighbor",
        "$R_6$\nmath",
    ]

    nan = np.nan
    # Cell values are SAT-producing rates where comparable. The labels carry
    # the exact denominator and timeout information used in the paper.
    values = np.array(
        [
            [1.00, 1.00, 0.00, nan, nan, nan, 1.00],
            [17 / 20, 707 / 893, nan, 43 / 893, 0.00, 15 / 20, nan],
            [1.00, 1.00, 1.00, nan, nan, nan, 1.00],
            [19 / 20, 19 / 20, 0.00, 0.00, nan, 1.00, 1.00],
            [23 / 50, nan, nan, 0.00, 0.00, nan, nan],
            [32 / 50, 32 / 50, nan, 0.00, 0.00, 0.00, nan],
        ],
        dtype=float,
    )
    labels = [
        ["SAT", "SAT", "UNSAT", "n/a", "n/a", "n/a", "SAT"],
        ["17/20\nSAT", "707/893\nSAT", "n/a", "43/893\nSAT\n3 TO", "0/893\nSAT", "15/20\nSAT", "n/a"],
        ["SAT", "SAT", "SAT", "n/a", "n/a", "n/a", "SAT"],
        ["19/20\nSAT", "19/20\nSAT", "0/20\nSAT", "0/20\nSAT", "n/a", "20/20\nSAT", "20/20\nSAT"],
        ["23/50\nSAT\n23 TO", "n/a", "n/a", "0/50\nSAT", "0/50\nSAT", "n/a", "n/a"],
        ["32/50\nSAT", "32/50\nSAT", "n/a", "0/50\nSAT", "0/50\nSAT", "0/50\nSAT", "n/a"],
    ]

    # Print-first grayscale ramp. Exact counts are printed in each cell; tone is
    # only a secondary cue, which keeps the figure journal-like and accessible.
    cmap = LinearSegmentedColormap.from_list(
        "sat_rate_gray", ["#F7F7F7", "#BDBDBD", "#4D4D4D"]
    )
    cmap.set_bad("#FFFFFF")

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="x", pad=3)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#FFFFFF",
                        edgecolor="#B5B5B5",
                        hatch="////",
                        linewidth=0.5,
                    )
                )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = labels[i][j]
            if np.isnan(values[i, j]):
                color = "#555555"
            else:
                color = readable_text_color(cmap(values[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("SAT-producing rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "region_transition_heatmap.png", dpi=300)
    plt.close(fig)


def make_lard_audit_ratios() -> None:
    with open("external_lard_witness_audit.json", "r", encoding="utf-8") as handle:
        audit = json.load(handle)

    baseline = audit["baselines"]
    witnesses = audit["witness_audit"]
    regions = {
        "R1 broad": [w for w in witnesses if w["region"] == "R1_pixel_ball"],
        "R2 band": [w for w in witnesses if w["region"] == "R2_mean_band"],
    }

    metrics = [
        (
            "Per-pair\nviolation",
            "per_pair_violation_frac",
            baseline["test_per_pair_violation_frac_mean"],
        ),
        (
            "Nearest-real\n$L_2$",
            "nearest_test_sameclass_L2",
            baseline["test_sameclass_nn_L2_median"],
        ),
        ("Total\nvariation", "tv", baseline["test_tv_median"]),
    ]

    labels = list(regions)
    ratios = np.zeros((len(labels), len(metrics)))
    for i, label in enumerate(labels):
        rows = regions[label]
        for j, (_, key, base) in enumerate(metrics):
            mean = float(np.mean([r[key] for r in rows]))
            ratios[i, j] = mean / base

    x = np.arange(len(metrics))
    width = 0.34
    colors = ["#F2F2F2", "#BDBDBD"]
    hatches = ["", "////"]

    fig, ax = plt.subplots(figsize=(4.7, 2.8))
    for i, label in enumerate(labels):
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            ratios[i],
            width,
            label=label,
            color=colors[i],
            edgecolor=EDGE,
            linewidth=0.6,
            hatch=hatches[i],
        )
        for bar, ratio in zip(bars, ratios[i]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.06,
                f"{ratio:.1f}x",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.axhline(1.0, color=EDGE, linewidth=0.9, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(25, ratios.max() * 1.6))
    ax.set_ylabel("Mean witness score / real baseline", fontsize=8)
    ax.set_xticks(x, [m[0] for m in metrics])
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(
        -0.45,
        0.93,
        "real baseline",
        fontsize=7,
        color=EDGE,
        va="top",
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "lard_witness_audit_ratios.png", dpi=300)
    plt.close(fig)


def make_p3_status_bars() -> None:
    """Plot expanded P3 status counts from the released result JSON files."""
    with open("verification_full_sample_centered.json", "r", encoding="utf-8") as handle:
        mean_clip = json.load(handle)["P3_mean_clip"]["epsilon_0.05"]
    with open("verification_p3_local_pca_expanded.json", "r", encoding="utf-8") as handle:
        local_pca = json.load(handle)["results_by_alpha"]
    with open(
        "verification_p3_local_pca_expanded_alpha3_200.json", "r", encoding="utf-8"
    ) as handle:
        local_pca_alpha3 = json.load(handle)["results_by_alpha"]["alpha_3.0"]
    with open("p3_random_control_summary.json", "r", encoding="utf-8") as handle:
        random_control = json.load(handle)["comparison"]

    rows = [
        {
            "label": "$L_\\infty$\n$\\varepsilon=0.05$",
            "verified": int(mean_clip["verified"]),
            "sat": int(mean_clip["violated"]),
            "timeout": 0,
            "n": int(mean_clip["n_total"]),
            "note": "",
        },
        {
            "label": "Local PCA\n$1\\sigma$",
            "verified": int(local_pca["alpha_1.0"]["verified"]),
            "sat": int(local_pca["alpha_1.0"]["sat"]),
            "timeout": int(local_pca["alpha_1.0"]["timeout"]),
            "n": int(local_pca["alpha_1.0"]["sample_size"]),
            "note": "",
        },
        {
            "label": "Random 8D\n$1\\sigma$",
            "verified": int(random_control["alpha_1.0"]["random_verified_median"]),
            "sat": int(random_control["alpha_1.0"]["random_sat_median"]),
            "timeout": int(random_control["alpha_1.0"]["random_timeout_median"]),
            "n": int(random_control["alpha_1.0"]["n"]),
            "note": "3 seeds",
        },
        {
            "label": "Local PCA\n$2\\sigma$",
            "verified": int(local_pca["alpha_2.0"]["verified"]),
            "sat": int(local_pca["alpha_2.0"]["sat"]),
            "timeout": int(local_pca["alpha_2.0"]["timeout"]),
            "n": int(local_pca["alpha_2.0"]["sample_size"]),
            "note": "",
        },
        {
            "label": "Random 8D\n$2\\sigma$",
            "verified": int(random_control["alpha_2.0"]["random_verified_median"]),
            "sat": int(random_control["alpha_2.0"]["random_sat_median"]),
            "timeout": int(random_control["alpha_2.0"]["random_timeout_median"]),
            "n": int(random_control["alpha_2.0"]["n"]),
            "note": (
                f"{int(random_control['alpha_2.0']['random_verified_min'])}-"
                f"{int(random_control['alpha_2.0']['random_verified_max'])}"
            ),
        },
        {
            "label": "Local PCA\n$3\\sigma$",
            "verified": int(local_pca_alpha3["verified"]),
            "sat": int(local_pca_alpha3["sat"]),
            "timeout": int(local_pca_alpha3["timeout"]),
            "n": int(local_pca_alpha3["sample_size"]),
            "note": "",
        },
        {
            "label": "Random 8D\n$3\\sigma$",
            "verified": int(random_control["alpha_3.0"]["random_verified_median"]),
            "sat": int(random_control["alpha_3.0"]["random_sat_median"]),
            "timeout": int(random_control["alpha_3.0"]["random_timeout_median"]),
            "n": int(random_control["alpha_3.0"]["n"]),
            "note": (
                f"{int(random_control['alpha_3.0']['random_verified_min'])}-"
                f"{int(random_control['alpha_3.0']['random_verified_max'])}"
            ),
        },
    ]

    for row in rows:
        total = row["verified"] + row["sat"] + row["timeout"]
        if total != row["n"]:
            raise ValueError(f"P3 status counts do not sum to n for {row['label']}: {total} != {row['n']}")

    labels = [row["label"] for row in rows]
    verified = np.array([row["verified"] for row in rows])
    sat = np.array([row["sat"] for row in rows])
    timeout = np.array([row["timeout"] for row in rows])
    totals = np.array([row["n"] for row in rows])

    colors = {
        "verified": "#F2F2F2",
        "sat": "#6E6E6E",
        "timeout": "#BDBDBD",
    }
    hatches = {
        "verified": "",
        "sat": "////",
        "timeout": "xx",
    }

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    y = np.arange(len(rows))
    ax.barh(
        y,
        verified,
        color=colors["verified"],
        edgecolor=EDGE,
        linewidth=0.5,
        hatch=hatches["verified"],
        label="UNSAT / verified",
    )
    ax.barh(
        y,
        sat,
        left=verified,
        color=colors["sat"],
        edgecolor=EDGE,
        linewidth=0.5,
        hatch=hatches["sat"],
        label="SAT witness",
    )
    ax.barh(
        y,
        timeout,
        left=verified + sat,
        color=colors["timeout"],
        edgecolor=EDGE,
        linewidth=0.5,
        hatch=hatches["timeout"],
        label="timeout",
    )

    for i, row in enumerate(rows):
        segments = [
            (row["verified"], 0, colors["verified"]),
            (row["sat"], row["verified"], colors["sat"]),
            (row["timeout"], row["verified"] + row["sat"], colors["timeout"]),
        ]
        for value, left, color in segments:
            if value == 0:
                continue
            x = left + value / 2
            text_color = readable_text_color(matplotlib.colors.to_rgba(color))
            label = str(value) if value >= max(12, row["n"] * 0.04) else ""
            if label:
                ax.text(x, i, label, ha="center", va="center", fontsize=8, color=text_color)
        side_label = f"n={row['n']}"
        if row["timeout"]:
            side_label += f"; TO {row['timeout']}"
        if row.get("note"):
            side_label += f"; {row['note']}"
        ax.text(
            row["n"] + max(totals) * 0.015,
            i,
            side_label,
            va="center",
            ha="left",
            fontsize=7,
            color="#333333",
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of centered anchors (random rows show median over three seeds)", fontsize=8)
    ax.set_xlim(0, max(totals) * 1.10)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "p3_status_bars.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    make_region_transition_heatmap()
    make_p3_status_bars()
    make_lard_audit_ratios()


if __name__ == "__main__":
    main()

"""
Assemble a side-by-side FC AE vs Conv AE comparison from:
  autoencoder_support_audit.json     (FC AE audit)
  verification_conv_ae.json          (conv AE audit + verdicts)
  verification_ae_with_mean_constraint.json  (FC AE verdicts)

Outputs:
  conv_vs_fc_comparison.json    (machine-readable)
  conv_vs_fc_comparison.txt     (LaTeX-ready table content)
"""
import json
import os


def fc_audit():
    d = json.load(open("autoencoder_support_audit.json"))
    out = {}
    for sub, s in d["subsets"].items():
        recon = s["all_subset_reconstruction"]["recon_linf"]
        op = s["tinytaxinet_output_preservation"]
        cs = s["classifier_scores"]
        out[sub] = {
            "n_images": s["n_images"],
            "recon_linf_median": recon["median"],
            "recon_linf_p95": recon["p95"],
            "recon_linf_max": recon["max"],
            "delta_cte_median": op["abs_delta_cte"]["median"],
            "delta_cte_p95": op["abs_delta_cte"]["p95"],
            "delta_he_median": op["abs_delta_he"]["median"],
            "delta_he_p95": op["abs_delta_he"]["p95"],
            "random_latent_clf_accept": cs["random_latent_decode_fraction_prob_gt_0.5"],
        }
    return out


def conv_audit():
    if not os.path.exists("verification_conv_ae.json"):
        return {}, {}
    d = json.load(open("verification_conv_ae.json"))
    audit = {}
    for sub, s in d["support_audit"].items():
        op = s["ttn_output_preservation"]
        audit[sub] = {
            "held_out_latent_coverage": s["held_out_latent_coverage"],
            "recon_linf_median": s["recon_linf_median"],
            "recon_linf_p95": s["recon_linf_p95"],
            "recon_linf_max": s["recon_linf_max"],
            "delta_cte_median": op["delta_CTE_median"],
            "delta_cte_p95": op["delta_CTE_p95"],
            "delta_he_median": op["delta_HE_median"],
            "delta_he_p95": op["delta_HE_p95"],
            "random_latent_clf_accept": s["classifier_random_latent_accept_rate"],
        }
    verdicts = d.get("verdicts", {})
    return audit, verdicts


def fc_verdicts():
    if not os.path.exists("verification_ae_with_mean_constraint.json"):
        return {}
    d = json.load(open("verification_ae_with_mean_constraint.json"))
    out = {}
    name_map = {
        "P1a_CTE_upper": "P1a",
        "P1b_CTE_lower": "P1b",
        "P4_C_left": "P4_left",
        "P4_C_right": "P4_right",
        "P5_deadzone": "P5",
    }
    for k, v in d.items():
        nice = name_map.get(k, k)
        r = v.get("ae_with_mean", {})
        out[nice] = {
            "status": r.get("status"),
            "time_seconds": r.get("time_seconds"),
            "CTE": r.get("counterexample", {}).get("CTE") if r.get("counterexample") else None,
        }
    return out


def fmt_n(v, w=6, prec=3):
    if v is None:
        return "{:>{w}s}".format("-", w=w)
    return "{:{w}.{p}f}".format(v, w=w, p=prec)


def main():
    fc_a = fc_audit()
    conv_a, conv_v = conv_audit()
    fc_v = fc_verdicts()

    out = {
        "fc_ae_audit": fc_a,
        "conv_ae_audit": conv_a,
        "fc_ae_verdicts": fc_v,
        "conv_ae_verdicts": conv_v,
    }
    with open("conv_vs_fc_comparison.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    lines = []
    lines.append("=== Per-subset AE support audit: FC AE vs Conv AE ===")
    lines.append(f"{'Subset':12s} {'arch':6s} {'recon med':>10s} {'recon p95':>10s} "
                 f"{'recon max':>10s} {'|dCTE| med':>10s} {'|dCTE| p95':>10s} "
                 f"{'rand-lat clf':>13s}")
    for sub in ["centered", "left", "right", "off_center"]:
        for arch, a in [("FC", fc_a.get(sub)), ("Conv", conv_a.get(sub))]:
            if a is None:
                lines.append(f"{sub:12s} {arch:6s} {'(missing)':>10s}")
                continue
            lines.append(f"{sub:12s} {arch:6s} "
                         f"{fmt_n(a['recon_linf_median'], 10)} "
                         f"{fmt_n(a['recon_linf_p95'], 10)} "
                         f"{fmt_n(a['recon_linf_max'], 10)} "
                         f"{fmt_n(a['delta_cte_median'], 10)} "
                         f"{fmt_n(a['delta_cte_p95'], 10)} "
                         f"{fmt_n(a['random_latent_clf_accept'], 13)}")
    lines.append("")
    lines.append("=== AE-composed verdicts (mean=0.5 enforced): FC AE vs Conv AE ===")
    lines.append(f"{'Query':10s} {'FC status':>12s} {'FC CTE':>10s} "
                 f"{'Conv status':>14s} {'Conv CTE':>10s}")
    for q in ["P1a", "P1b", "P4_left", "P4_right", "P5"]:
        fv = fc_v.get(q, {})
        cv = conv_v.get(q, {})
        cte_fc = fv.get("CTE")
        cte_cv = cv.get("CTE")
        lines.append(f"{q:10s} {fv.get('status','-'):>12s} "
                     f"{'-' if cte_fc is None else f'{cte_fc:>10.3f}'} "
                     f"{cv.get('status','-'):>14s} "
                     f"{'-' if cte_cv is None else f'{cte_cv:>10.3f}'}")
    text = "\n".join(lines)
    with open("conv_vs_fc_comparison.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()

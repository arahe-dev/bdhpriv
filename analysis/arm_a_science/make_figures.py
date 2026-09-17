"""Paper-quality figures for the Arm-A sparse-population science mission.

Reads the JSON/NPZ results and writes PNGs to figures/arm_a_science/ and
figures/sae/. No new measurements are made here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import FIG_DIR, LADDER, RESULTS_DIR  # noqa: E402

SAE_FIG = Path(__file__).resolve().parents[2] / "figures" / "sae"
SAE_RES = Path(__file__).resolve().parents[2] / "results" / "sae"
CKPT_ORDER = ["random_init", "step2000", "step18000", "step19000", "latest"]
CKPT_LABEL = {"random_init": "init", "step2000": "2k", "step18000": "18k",
              "step19000": "19k", "latest": "19.1k"}


def _load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def fig1_training():
    mat = _load("sparsity_maturation.json")["checkpoints"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=140)
    xs = [mat[c]["step"] for c in CKPT_ORDER]
    for key, color in (("x", "tab:blue"), ("y", "tab:green"),
                       ("u", "tab:red")):
        z = [mat[c]["zero_fraction"][key]["mean"] for c in CKPT_ORDER]
        pz = [mat[c]["pair_zero_fraction"][key]["mean"] for c in CKPT_ORDER]
        axes[0, 0].plot(xs, z, "o-", color=color, label=f"{key}")
        axes[0, 1].plot(xs, pz, "o-", color=color, label=f"{key} pair")
    axes[0, 0].set_title("exact zero fraction")
    axes[0, 1].set_title("RoPE-pair zero fraction")
    for key, color in (("x", "tab:blue"), ("u", "tab:red")):
        neff = [mat[c]["concentration"][key]["neff_over_K"]["mean"]
                for c in CKPT_ORDER]
        gini = [mat[c]["concentration"][key]["gini"]["mean"]
                for c in CKPT_ORDER]
        top5 = [mat[c]["concentration"][key]["top5pct_mass_share"]["mean"]
                for c in CKPT_ORDER]
        axes[1, 0].plot(xs, neff, "o-", color=color, label=f"{key}")
        axes[1, 1].plot(xs, gini, "o-", color=color, label=f"{key} Gini")
        axes[1, 1].plot(xs, top5, "s--", color=color,
                        label=f"{key} top5% mass")
    axes[1, 0].set_title("effective population N_eff / K")
    axes[1, 1].set_title("inequality")
    for ax in axes.ravel():
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes[:, 1]:
        ax.set_xlabel("training step")
    fig.suptitle("FIG 1  Arm-A population maturation (synthetic packed "
                 "batches, one trajectory)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_training_evolution.png")
    plt.close(fig)


def fig2_local_global():
    topn = _load("topn_global_local.json")["checkpoints"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=140)
    for ax, key in zip(axes[:2], ("x", "u")):
        res = topn["latest"]["per_cell"][key]["levels"]
        loc = np.array([[res[l]["heads"][h]["local"][str(n)]["mean"]
                         for n in LADDER] for l in range(len(res))
                        for h in range(len(res[l]["heads"]))])
        glo = np.array([[res[l]["heads"][h]["global"][str(n)]["mean"]
                         for n in LADDER] for l in range(len(res))
                        for h in range(len(res[l]["heads"]))])
        x = np.array(LADDER, dtype=np.float64)
        ax.plot(x, loc.mean(0), "o-", label="M_local (own top-N)")
        ax.fill_between(x, np.quantile(loc, 0.25, axis=0),
                        np.quantile(loc, 0.75, axis=0), alpha=0.2)
        ax.plot(x, glo.mean(0), "s-", label="M_global (frozen global top-N)")
        ax.fill_between(x, np.quantile(glo, 0.25, axis=0),
                        np.quantile(glo, 0.75, axis=0), alpha=0.2)
        ax.plot(x, x / 4096.0, "k--", lw=1, label="uniform N/K")
        ax.set_xscale("log", base=2)
        ax.set_xticks(LADDER)
        ax.set_xticklabels([str(n) for n in LADDER], rotation=45, fontsize=7)
        ax.set_xlabel("N (coordinates per head)")
        ax.set_ylabel("fraction of token mass")
        ax.set_title(f"{key}: local vs global top-N")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    # pooled x and u
    ax = axes[2]
    for key, color in (("x", "tab:blue"), ("u", "tab:red")):
        res = topn["latest"]["pooled"][key]["levels"]
        loc = np.array([[res[l]["local"][str(n)]["mean"] for n in LADDER]
                        for l in range(len(res))])
        glo = np.array([[res[l]["global"][str(n)]["mean"] for n in LADDER]
                        for l in range(len(res))])
        ax.plot(LADDER, loc.mean(0), "o-", color=color,
                label=f"{key} M_local")
        ax.plot(LADDER, glo.mean(0), "s--", color=color,
                label=f"{key} M_global")
    ax.plot(LADDER, np.array(LADDER) / 16384, "k--", lw=1,
            label="uniform N/16384")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LADDER)
    ax.set_xticklabels([str(n) for n in LADDER], rotation=45, fontsize=7)
    ax.set_xlabel("N (coordinates, all heads pooled)")
    ax.set_title("pooled over 4 heads")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.suptitle("FIG 2  Global vs conditional concentration, cross-fitted "
                 "(E1/E2 synthetic packed batches)")
    fig.tight_layout()
    fig.savefig(FIG_A() / "fig2_local_vs_global_topn.png")
    plt.close(fig)


def FIG_A():
    return FIG_DIR


def fig3_stability():
    stab = _load("population_stability.json")["checkpoints"]["latest"][
        "summary"]
    lags = [1, 2, 4, 8, 16, 32, 63, 127]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    for ax, key in zip(axes, ("x", "u")):
        s = stab[key]
        for n, style in (("64", "o-"), ("256", "s-"), ("1024", "^-")):
            y = [s[f"lag_{l}"]["jaccard_mean_per_N"][n] for l in lags]
            ax.plot(lags, y, style, label=f"Jaccard top-{n}")
            ax.axhline(s["cross_row"]["jaccard_mean_per_N"][n],
                       ls=":", lw=1, alpha=0.5)
        ax.plot(lags, [s[f"lag_{l}"]["support_jaccard_mean"] for l in lags],
                "d-", color="gray", label="support Jaccard")
        ax.plot(lags, [s[f"lag_{l}"]["spearman_mean"] for l in lags],
                "v-", color="tab:green", label="Spearman rho")
        ax.plot(lags, [s[f"lag_{l}"]["rbo_mean"] for l in lags],
                "x-", color="tab:orange", label="RBO")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lags)
        ax.set_xticklabels([str(l) for l in lags])
        ax.set_xlabel("token lag within document")
        ax.set_ylabel("overlap")
        ax.set_ylim(0, 1)
        ax.set_title(f"{key}: population overlap vs distance")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    fig.suptitle("FIG 3  Top-N identity rotation and rank stability "
                 "(latest, synthetic packed batches)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_population_stability.png")
    plt.close(fig)


def fig4_core_tail():
    core = _load("core_tail.json")["checkpoints"]["latest"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=140)
    defs = ["mass_top_1pct", "mass_top_6p25", "mass_top_25"]
    sizes = [0.01, 0.0625, 0.25]
    for key, color in (("x", "tab:blue"), ("u", "tab:red")):
        means, lo, hi = [], [], []
        for name in defs:
            ents = [l for e in core["defs"][name]["levels"]
                    if e["key"] == key for l in e["levels"]]
            v = np.array([l["mass_fraction"]["mean"] for l in ents])
            means.append(v.mean())
            lo.append(np.quantile(v, 0.1))
            hi.append(np.quantile(v, 0.9))
        axes[0].errorbar(np.array(sizes) * 100, means,
                         yerr=[np.array(means) - np.array(lo),
                               np.array(hi) - np.array(means)],
                         fmt="o-", color=color, label=f"{key} core mass share",
                         capsize=3)
        # residual top-N after removing 6.25% core
        ents = [l for e in core["defs"]["mass_top_6p25"]["levels"]
                if e["key"] == key for l in e["levels"]]
        res = np.array([[l["residual_topN"][str(n)]["mean"] for n in LADDER]
                        for l in ents])
        axes[1].plot(LADDER, res.mean(0), "o-", color=color,
                     label=f"{key} residual top-N")
        need50 = np.array([l["residual_need_coords"]["p50"]["mean"]
                           for l in ents])
        axes[2].plot(range(len(need50)), need50, "o-", color=color,
                     label=f"{key}: coords for 50% residual")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("global core size (% of K)")
    axes[0].set_ylabel("fraction of token mass in core")
    axes[0].set_title("core mass fraction")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(LADDER)
    axes[1].set_xticklabels([str(n) for n in LADDER], rotation=45, fontsize=7)
    axes[1].set_xlabel("N (non-core coordinates)")
    axes[1].set_ylabel("fraction of residual mass")
    axes[1].set_title("residual (tail) top-N after 6.25% core")
    axes[1].legend(fontsize=7)
    axes[2].set_xlabel("level")
    axes[2].set_ylabel("coordinates (of 3840 non-core)")
    axes[2].set_title("tail width for 50% of residual")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("FIG 4  Core + conditional tail decomposition (latest)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_core_tail.png")
    plt.close(fig)


def fig5_frequency():
    fb = _load("frequency_bands.json")["checkpoints"]["latest"]
    fn = _load("frequency_null.json")["checkpoints"]["latest"][
        "band_null_intervals"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=140)
    bands = np.arange(8)
    for ax, key in zip(axes[:2], ("x", "u")):
        # per-head profiles
        for h in range(4):
            prof = np.mean([fb[key]["levels"][l]["heads"][h]["bands8"][
                "mass_share"] for l in range(8)], axis=0)
            ax.plot(bands, prof, "o-", alpha=0.6, label=f"head {h}")
        pool = np.array(fn[key]["bands8"]["observed_mass_share"])
        lo = np.array(fn[key]["bands8"]["mass_share_null_p2.5"])
        hi = np.array(fn[key]["bands8"]["mass_share_null_p97.5"])
        ax.fill_between(bands, lo, hi, color="k", alpha=0.15,
                        label="pair-preserving null 95%")
        ax.plot(bands, pool, "k^-", label="pooled observed")
        ax.set_xticks(bands)
        ax.set_xticklabels([f"b{i}" for i in range(8)])
        ax.set_xlabel("band (b0 fastest -> b7 slowest)")
        ax.set_ylabel("mass share")
        ax.set_title(f"{key}: band mass share, per head")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    argmax = np.zeros(8)
    cnt = 0
    for l in range(8):
        for h in range(4):
            prof = np.array(fb["x"]["levels"][l]["heads"][h]["bands8"][
                "mass_share"])
            argmax[prof.argmax()] += 1
    axes[2].bar(bands, argmax / argmax.sum())
    axes[2].set_xticks(bands)
    axes[2].set_xlabel("band")
    axes[2].set_ylabel("fraction of level-head cells")
    axes[2].set_title("x: which band dominates each cell")
    axes[2].grid(alpha=0.25)
    fig.suptitle("FIG 5  RoPE frequency organization with pair-preserving "
                 "null (latest, 5000 permutations)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_frequency_bands.png")
    plt.close(fig)


def fig6_gemma():
    small = json.loads((SAE_RES / "gemma_small.json").read_text(
        encoding="utf-8"))
    medium = json.loads((SAE_RES / "gemma_medium.json").read_text(
        encoding="utf-8"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=140)
    for d, label, color in ((small, "L0=20", "tab:blue"),
                            (medium, "L0=60", "tab:red")):
        q = np.array([v for k, v in d["norms"]["encoder_direction"].items()
                      if k not in ("mean", "std", "min", "max", "n")])
        x = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9])
        axes[0].plot(x, q, "o-", color=color, label=label)
        qt = np.array([v for k, v in d["threshold"]["quantiles"].items()
                       if k not in ("mean", "std", "min", "max", "n")])
        axes[1].plot(x, qt, "o-", color=color, label=label)
        qn = np.array([v for k, v in d["decoder_similarity"]["nn1_top1"].items()
                       if k not in ("mean", "std", "min", "max", "n")])
        axes[2].plot(x, qn, "o-", color=color, label=label)
        qe = np.array([v for k, v in
                       d["encoder_similarity"]["nn1_top1"].items()
                       if k not in ("mean", "std", "min", "max", "n")])
        axes[3].plot(x, qe, "o-", color=color, label=label)
    axes[0].set_title("encoder direction norm")
    axes[1].set_title("jump_relu threshold")
    axes[2].set_title("decoder NN cosine (top-1)")
    axes[3].set_title("encoder NN cosine (top-1)")
    for ax in axes:
        ax.set_xscale("logit" if False else "linear")
        ax.set_xlabel("percentile")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("FIG 6  Gemma Scope 2 270M l12 SAE structure, L0=20 vs L0=60")
    fig.tight_layout()
    fig.savefig(SAE_FIG / "fig6_gemma_small_vs_medium.png")
    plt.close(fig)


def fig7_cross_system():
    d = json.loads((SAE_RES / "cross_system_comparison.json").read_text(
        encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)
    entries = [
        ("Arm-A x mass (pooled)", d["systems"]["arm_a_latest"][
            "x_mass_pooled"], "tab:blue", "-"),
        ("Arm-A u mass (pooled)", d["systems"]["arm_a_latest"][
            "u_mass_pooled"], "tab:red", "-"),
        ("Gemma enc norm^2 (small)", d["systems"]["gemma_small"][
            "encoder_norm_sq"], "tab:green", "--"),
        ("Anthropic density", d["systems"]["anthropic_public"]["density"],
         "tab:purple", "-."),
    ]
    for label, e, color, style in entries:
        cur = e["top_share_curve"]
        xs = sorted(float(k) for k in cur)
        ys = [cur[str(k)] if str(k) in cur else cur[f"{k:g}"] for k in xs]
        axes[0].plot(xs, ys, style, marker="o", color=color, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("fraction of units (log)")
    axes[0].set_ylabel("fraction of total importance")
    axes[0].set_title("concentration curves (dimensionless)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    labels = [e[0] for e in entries]
    ginis = [e[1]["gini"] for e in entries]
    neffs = [e[1]["neff_over_n"] for e in entries]
    xpos = np.arange(len(entries))
    axes[1].bar(xpos - 0.2, ginis, width=0.4, label="Gini")
    axes[1].bar(xpos + 0.2, neffs, width=0.4, label="N_eff / N")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([l.replace(" (", "\n(") for l in labels],
                            fontsize=6.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("inequality summaries (same denominator per entry)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("FIG 7  Normalized Arm-A vs external sparse representations")
    fig.tight_layout()
    fig.savefig(SAE_FIG / "fig7_cross_system.png")
    plt.close(fig)


def fig8_context_length():
    d = _load("context_length.json")["by_T"]
    Ts = sorted(int(t) for t in d)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=140)
    for key, N, color, style in (("u", 64, "tab:red", "o-"),
                                 ("x", 16, "tab:blue", "s-"),
                                 ("x", 1024, "tab:blue", "s--")):
        ys = [d[str(T)][f"delta_{key}_{N}"]["mean_over_cells"] for T in Ts]
        axes[0].plot(Ts, ys, style, color=color,
                     label=f"Delta_{key}({N})")
    for key, color in (("u", "tab:red"), ("x", "tab:blue")):
        ys = [d[str(T)][f"core_6p25_mass_frac_{key}"]["mean"] for T in Ts]
        axes[1].plot(Ts, ys, "o-", color=color, label=f"core {key}")
        zs = [d[str(T)]["bands"][key]["z"] for T in Ts]
        axes[2].plot(Ts, zs, "o-", color=color, label=f"band z {key}")
        j1 = [d[str(T)]["stability"][key]["lag_1"]["jaccard64_mean"]
              for T in Ts]
        jc = [d[str(T)]["stability"][key]["cross_row"]["jaccard64_mean"]
              for T in Ts]
        axes[3].plot(Ts, j1, "o-", color=color, label=f"{key} lag-1")
        axes[3].plot(Ts, jc, "s--", color=color, label=f"{key} cross-doc")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(T) for T in Ts])
        ax.set_xlabel("evaluation context T")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    axes[0].set_title("conditional gap vs T")
    axes[1].set_title("core (6.25%) mass fraction")
    axes[2].set_title("pooled band structure z")
    axes[3].set_title("top-64 stability vs T")
    fig.suptitle("FIG 8  Context-length robustness of the core+tail structure "
                 "(latest, synthetic packed batches)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_context_length.png")
    plt.close(fig)


def fig9_recurrence_probe():
    d = _load("semantic_stability.json")
    summary = d["summary"]
    lags = (256, 512, 1024)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    for key, color in (("x", "tab:blue"), ("y", "tab:green"),
                       ("u", "tab:red")):
        r = [summary[f"repeat|repeat_lag_{l}|{key}"]["jaccard64_mean"]
             for l in lags]
        s = [summary[f"shuffle|shuffle_lag_{l}|{key}"]["jaccard64_mean"]
             for l in lags]
        axes[0].plot(lags, r, "o-", color=color, label=f"{key} repeated")
        axes[0].plot(lags, s, "s--", color=color, label=f"{key} shuffled")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(list(lags))
    axes[0].set_xticklabels([str(l) for l in lags])
    axes[0].set_xlabel("repetition lag (tokens)")
    axes[0].set_ylabel("top-64 Jaccard")
    axes[0].set_ylim(0, 0.8)
    axes[0].set_title("aligned repeated positions vs matched shuffled control")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    for key, color in (("x", "tab:blue"), ("y", "tab:green"),
                       ("u", "tab:red")):
        rr = summary[f"repeat|repeat_lag_256|{key}"]["level_mean_jaccard64"]
        ss = summary[f"shuffle|shuffle_lag_256|{key}"]["level_mean_jaccard64"]
        lv = sorted(int(k) for k in rr)
        axes[1].plot(lv, [rr[str(l)] for l in lv], "o-", color=color,
                     label=f"{key} repeated")
        axes[1].plot(lv, [ss[str(l)] for l in lv], "s--", color=color,
                     label=f"{key} shuffled")
    axes[1].set_xlabel("level")
    axes[1].set_ylabel("top-64 Jaccard at lag 256")
    axes[1].set_title("per-level persistence (level 0 x is token-identity)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("FIG 9  Structured-recurrence probe: the conditional tail "
                 "tracks content, not position (E2 synthetic)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_recurrence_probe.png")
    plt.close(fig)


def fig10_forward_ablation():
    path = RESULTS_DIR / "e4_forward_ablation.json"
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))["results"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    widths = [320, 512, 1024]
    for strat, label, color in (("static", "static (global top-w)",
                                 "tab:red"),
                                ("hybrid", "hybrid (256 core + tail)",
                                 "tab:orange"),
                                ("dynamic", "dynamic (oracle top-w)",
                                 "tab:blue")):
        ys = []
        for w in widths:
            key = f"{strat}_{w}" if strat != "hybrid" else {
                320: "hybrid_256p64", 512: "hybrid_256p256",
                1024: "hybrid_256p768"}[w]
            ys.append(d[key]["relative_logit_perturbation_mean"])
        axes[0].plot(widths, ys, "o-", color=color, label=label)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(widths)
    axes[0].set_xticklabels([str(w) for w in widths])
    axes[0].set_xlabel("active coordinates per (level, head)")
    axes[0].set_ylabel("relative logit perturbation")
    axes[0].set_title("forward-only ablation (oracle selection)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    ks = [d[k]["u_mass_kept"] for k in
          ("static_320", "hybrid_256p64", "dynamic_320",
           "static_512", "hybrid_256p256", "dynamic_512",
           "static_1024", "hybrid_256p768", "dynamic_1024")]
    ps = [d[k]["relative_logit_perturbation_mean"] for k in
          ("static_320", "hybrid_256p64", "dynamic_320",
           "static_512", "hybrid_256p256", "dynamic_512",
           "static_1024", "hybrid_256p768", "dynamic_1024")]
    colors = ["tab:red", "tab:orange", "tab:blue"] * 3
    axes[1].scatter(ks, ps, c=colors, s=40)
    for k, x, y in zip(("static_320", "hybrid_256p64", "dynamic_320",
                        "static_512", "hybrid_256p256", "dynamic_512",
                        "static_1024", "hybrid_256p768", "dynamic_1024"),
                       ks, ps):
        axes[1].annotate(k, (x, y), fontsize=6)
    axes[1].set_xlabel("u mass kept")
    axes[1].set_ylabel("relative logit perturbation")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_title("fidelity vs mass coverage")
    axes[1].grid(alpha=0.25)
    fig.suptitle("FIG 10  E4 pilot: oracle static/dynamic/hybrid u-masking "
                 "(forward-only; not training evidence)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_forward_ablation.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SAE_FIG.mkdir(parents=True, exist_ok=True)
    fig1_training()
    fig2_local_global()
    fig3_stability()
    fig4_core_tail()
    fig5_frequency()
    fig6_gemma()
    fig7_cross_system()
    fig8_context_length()
    fig9_recurrence_probe()
    fig10_forward_ablation()
    print(json.dumps({"figures": [str(p.name) for p in
                                  sorted(FIG_DIR.glob("*.png"))] +
                                 [str(p.name) for p in
                                  sorted(SAE_FIG.glob("*.png"))]}, indent=2))


if __name__ == "__main__":
    main()

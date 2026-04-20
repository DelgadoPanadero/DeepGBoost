"""
Analysis: H+I config (10L×10T, lr=0.3, max_features="sqrt") vs historical baseline.

Compares current dev suite results against the per_class_trees baseline from
BITACORA Experimento 6 (20L×5T, lr=0.1) using absolute score and DGB-XGB gap.

Outputs: summary tables to stdout, plots to /tmp/config_hi_*.png
"""

import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS = ["GradientBoosting", "RandomForest", "XGBoost", "DeepGBoost"]

# Historical baseline from BITACORA Experimento 6 (cross-validation, 5-fold, 1 run)
HISTORICAL_DGB = {
    "penguins":         0.9911,
    "bankmarketing":    0.8914,
    "abalone":          0.5483,
    "adult":            0.8487,
    "navalvessel":      0.9953,
    "californiahousing":0.8285,
    "bikesales":        0.8930,
    "concrete":         0.9354,
}
HISTORICAL_XGB = {
    "penguins":         0.9880,
    "bankmarketing":    0.9039,
    "abalone":          0.5383,
    "adult":            0.8691,
    "navalvessel":      0.9860,
    "californiahousing":0.8322,
    "bikesales":        0.8868,
    "concrete":         0.9320,
}

TASK = {
    "penguins": "clf", "bankmarketing": "clf", "abalone": "clf", "adult": "clf",
    "navalvessel": "reg", "californiahousing": "reg", "bikesales": "reg", "concrete": "reg",
}


def load_scores(dataset_name):
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_cross_validation_test_scores.json")
    folds = json.load(open(path))
    means, stds = {}, {}
    for m in MODELS:
        vals = [f[m] for f in folds if m in f]
        means[m] = statistics.mean(vals)
        stds[m] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return means, stds


def main():
    datasets = sorted([
        k.replace("_cross_validation_test_scores.json", "")
        for k in os.listdir(RESULTS_DIR)
        if k.endswith("_scores.json")
    ])

    rows = []
    for ds in datasets:
        means, stds = load_scores(ds)
        dgb_new = means["DeepGBoost"]
        xgb_new = means["XGBoost"]
        dgb_old = HISTORICAL_DGB.get(ds)
        xgb_old = HISTORICAL_XGB.get(ds)
        gap_new = dgb_new - xgb_new
        gap_old = (dgb_old - xgb_old) if dgb_old is not None else None
        delta_dgb = (dgb_new - dgb_old) if dgb_old is not None else None
        rows.append({
            "dataset": ds,
            "task": TASK.get(ds, "?"),
            "gb": means["GradientBoosting"],
            "rf": means["RandomForest"],
            "xgb_new": xgb_new,
            "dgb_new": dgb_new,
            "dgb_std": stds["DeepGBoost"],
            "gap_new": gap_new,
            "dgb_old": dgb_old,
            "gap_old": gap_old,
            "delta_dgb": delta_dgb,
        })

    # ── Table 1: Full results ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE 1 — Dev Suite Results (H+I config: 10L×10T, lr=0.3, max_features=sqrt)")
    print("=" * 90)
    print(f"{'Dataset':<22} {'Task':<5} {'GB':>8} {'RF':>8} {'XGB':>8} {'DGB':>8} {'±':>6} {'DGB-XGB':>9} {'Winner'}")
    print("-" * 90)
    for r in rows:
        flag = "🏆 DGB" if r["gap_new"] > 0.001 else ("▼ XGB" if r["gap_new"] < -0.005 else "≈ tie")
        print(f"{r['dataset']:<22} {r['task']:<5} {r['gb']:>8.4f} {r['rf']:>8.4f} "
              f"{r['xgb_new']:>8.4f} {r['dgb_new']:>8.4f} {r['dgb_std']:>6.4f} "
              f"{r['gap_new']:>+9.4f}  {flag}")

    # ── Table 2: Config comparison ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE 2 — Config comparison: H+I (new) vs per_class_trees baseline (old, BITACORA Exp.6)")
    print("=" * 90)
    print(f"{'Dataset':<22} {'Task':<5} {'DGB old':>9} {'DGB new':>9} {'ΔDGB':>8} {'Gap old':>9} {'Gap new':>9} {'ΔGap':>8}")
    print("-" * 90)
    for r in rows:
        if r["dgb_old"] is None:
            continue
        delta_gap = r["gap_new"] - r["gap_old"]
        trend = "▲" if r["delta_dgb"] > 0.001 else ("▼" if r["delta_dgb"] < -0.001 else "≈")
        print(f"{r['dataset']:<22} {r['task']:<5} {r['dgb_old']:>9.4f} {r['dgb_new']:>9.4f} "
              f"{r['delta_dgb']:>+8.4f}{trend} {r['gap_old']:>+9.4f} {r['gap_new']:>+9.4f} {delta_gap:>+8.4f}")

    # ── Aggregates ──────────────────────────────────────────────────────────
    clf_rows = [r for r in rows if r["task"] == "clf" and r["dgb_old"] is not None]
    reg_rows = [r for r in rows if r["task"] == "reg" and r["dgb_old"] is not None]

    print("\n" + "=" * 60)
    print("TABLE 3 — Mean DGB-XGB gap by task type")
    print("=" * 60)
    for label, group in [("Classification (4 datasets)", clf_rows), ("Regression (4 datasets)", reg_rows)]:
        mean_gap_old = statistics.mean(r["gap_old"] for r in group)
        mean_gap_new = statistics.mean(r["gap_new"] for r in group)
        wins_new = sum(1 for r in group if r["gap_new"] > 0.001)
        wins_old = sum(1 for r in group if r["gap_old"] > 0.001)
        print(f"\n  {label}")
        print(f"    Mean gap old : {mean_gap_old:+.4f}  ({wins_old}/4 wins vs XGB)")
        print(f"    Mean gap new : {mean_gap_new:+.4f}  ({wins_new}/4 wins vs XGB)")
        print(f"    Δ mean gap   : {mean_gap_new - mean_gap_old:+.4f}")

    # ── Plot: DGB-XGB gap old vs new ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DGB-XGB gap: H+I config vs baseline", fontsize=13, fontweight="bold")

    for ax, (label, group) in zip(axes, [("Classification", clf_rows), ("Regression", reg_rows)]):
        names = [r["dataset"] for r in group]
        gaps_old = [r["gap_old"] for r in group]
        gaps_new = [r["gap_new"] for r in group]
        x = np.arange(len(names))
        w = 0.35
        bars_old = ax.bar(x - w/2, gaps_old, w, label="Baseline (20L×5T, lr=0.1)", color="#4878d0", alpha=0.85)
        bars_new = ax.bar(x + w/2, gaps_new, w, label="H+I (10L×10T, lr=0.3)", color="#ee854a", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("DGB − XGB (higher = DGB wins)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/tmp/config_hi_gap_comparison.png", dpi=130)
    print("\nPlot saved: /tmp/config_hi_gap_comparison.png")

    # ── Plot: absolute DGB scores ────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("DeepGBoost absolute score: H+I config vs baseline", fontsize=13, fontweight="bold")

    for ax, (label, group) in zip(axes2, [("Classification", clf_rows), ("Regression", reg_rows)]):
        names = [r["dataset"] for r in group]
        scores_old = [r["dgb_old"] for r in group]
        scores_new = [r["dgb_new"] for r in group]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, scores_old, w, label="Baseline (20L×5T, lr=0.1)", color="#4878d0", alpha=0.85)
        ax.bar(x + w/2, scores_new, w, label="H+I (10L×10T, lr=0.3)", color="#ee854a", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Weighted F1 / R²")
        ax.legend(fontsize=8)
        ymin = min(min(scores_old), min(scores_new)) * 0.98
        ax.set_ylim(ymin, 1.02)

    plt.tight_layout()
    plt.savefig("/tmp/config_hi_scores_comparison.png", dpi=130)
    print("Plot saved: /tmp/config_hi_scores_comparison.png")


if __name__ == "__main__":
    main()

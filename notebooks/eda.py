"""
notebooks/eda.py
================
Exploratory data analysis — run this after the sweep to understand your dataset
before training ML models. Generates a full set of diagnostic figures.

Run as: python notebooks/eda.py
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing.preprocess import load_results, clean_data, engineer_features
from data_collection.parameter_sweep import PARAM_NAMES
from config.settings import RESULTS_DIR, ML_TARGETS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FIG_DIR = RESULTS_DIR / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("husl")


def run_eda():
    df_raw = load_results()
    log.info(f"Raw dataset: {df_raw.shape}")

    df = clean_data(df_raw)
    df = engineer_features(df)
    log.info(f"Clean dataset: {df.shape}")

    # 1. Target distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, target in zip(axes.flatten(), ML_TARGETS):
        if target in df.columns:
            df[target].hist(ax=ax, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
            ax.axvline(df[target].mean(), color="red", ls="--", label=f"Mean={df[target].mean():.2f}")
            ax.set_title(target)
            ax.set_xlabel(target)
            ax.legend(fontsize=8)
    plt.suptitle("Target Variable Distributions", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "target_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: target_distributions.png")

    # 2. Parameter vs lap_time scatter (most important relationship)
    if "lap_time_s" not in df.columns:
        log.warning("lap_time_s not in dataset")
        return

    n_params = len(PARAM_NAMES)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()
    for ax, param in zip(axes, PARAM_NAMES):
        if param in df.columns:
            ax.scatter(df[param], df["lap_time_s"], alpha=0.3, s=15, edgecolors="none")
            # Trend line
            z = np.polyfit(df[param], df["lap_time_s"], 1)
            xline = np.linspace(df[param].min(), df[param].max(), 100)
            ax.plot(xline, np.polyval(z, xline), "r-", lw=1.5)
            corr = df[param].corr(df["lap_time_s"])
            ax.set_xlabel(param, fontsize=8)
            ax.set_ylabel("Lap time (s)", fontsize=7)
            ax.set_title(f"r = {corr:.3f}", fontsize=9)
            ax.tick_params(labelsize=6)
    for ax in axes[n_params:]:
        ax.set_visible(False)
    plt.suptitle("Parameter vs Lap Time (with linear trend)", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "params_vs_laptime.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: params_vs_laptime.png")

    # 3. Correlation heatmap
    param_cols = [p for p in PARAM_NAMES if p in df.columns]
    target_cols = [t for t in ML_TARGETS if t in df.columns]
    corr_matrix = df[param_cols + target_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr_matrix, ax=ax, mask=mask,
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=False, linewidths=0.3, square=False)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: correlation_heatmap.png")

    # 4. Cross-correlations between targets
    fig, ax = plt.subplots(figsize=(8, 6))
    target_corr = df[target_cols].corr()
    sns.heatmap(target_corr, ax=ax, cmap="coolwarm", center=0,
                annot=True, fmt=".3f", vmin=-1, vmax=1, square=True)
    ax.set_title("Target Cross-Correlation", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "target_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: target_correlation.png")

    # 5. Spring balance vs lap time (key engineered feature)
    if "spring_balance" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(df["spring_balance"], df["lap_time_s"],
                        c=df["total_downforce"], cmap="RdYlGn_r", alpha=0.5, s=20)
        axes[0].set_xlabel("Spring Balance (front-rear / total)")
        axes[0].set_ylabel("Lap Time (s)")
        axes[0].set_title("Spring Balance vs Lap Time\n(colour = total downforce)")
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r")
        plt.colorbar(sm, ax=axes[0], label="Total Downforce")

        # Platform stiffness vs corner efficiency
        axes[1].scatter(df["platform_stiffness"], df["corner_efficiency"],
                        c=df["lap_time_s"], cmap="RdYlGn_r", alpha=0.5, s=20)
        axes[1].set_xlabel("Platform Stiffness Index")
        axes[1].set_ylabel("Corner Efficiency")
        axes[1].set_title("Platform Stiffness vs Corner Efficiency\n(colour = lap time)")
        plt.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn_r"), ax=axes[1], label="Lap Time (s)")

        plt.tight_layout()
        fig.savefig(FIG_DIR / "engineered_features.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved: engineered_features.png")

    # 6. Summary statistics
    stats = df[param_cols + target_cols].describe().round(4)
    stats.to_csv(RESULTS_DIR / "dataset_statistics.csv")
    log.info(f"\nDataset statistics:\n{stats[target_cols]}")

    log.info(f"\nAll EDA figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
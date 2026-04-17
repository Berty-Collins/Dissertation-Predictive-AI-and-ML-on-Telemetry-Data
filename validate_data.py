"""
validate_data.py
================
Full data-quality check for results/sweep_results.csv before ML training.

Checks:
  1.  Load & completeness  -- expected 335 runs, source breakdown
  2.  Schema               -- required param/KPI columns present
  3.  NaN audit
  4.  Dead channels        -- zero or near-zero variance KPIs
  5.  Outlier flagging     -- IQR-based per KPI column
  6.  Parameter distributions & coverage
  7.  KPI distributions    -- histograms + summary stats
  8.  Param->KPI correlation heatmap
  9.  OAT sanity check     -- each param should show a trend
  10. LHS space-fill check
  11. Pairplot sample
  12. Damage / crash audit
  13. Export clean_data.csv with _flag_outlier column

Run:
    python validate_data.py

Outputs:
    results/validation_report.txt
    results/plots/
    results/clean_data.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
while not (ROOT / "config").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
from config.settings import (
    RESULTS_DIR, PARAM_COLS, ML_TARGETS,
    EXPECTED_RUNS, EXPECTED_SOURCES,
)

PLOTS_DIR  = RESULTS_DIR / "plots"
CSV_IN     = RESULTS_DIR / "sweep_results.csv"
CSV_OUT    = RESULTS_DIR / "clean_data.csv"
REPORT_OUT = RESULTS_DIR / "validation_report.txt"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

def section(title):
    bar = "=" * 70
    log(); log(bar); log(f"  {title}"); log(bar)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
section("1. LOAD")

if not CSV_IN.exists():
    log(f"ERROR: {CSV_IN} not found. Run scenario_runner.py first.")
    sys.exit(1)

df = pd.read_csv(CSV_IN)
log(f"Loaded {len(df)} rows x {len(df.columns)} columns")
log(f"Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPLETENESS
# ─────────────────────────────────────────────────────────────────────────────
section("2. COMPLETENESS")

log(f"Expected runs : {EXPECTED_RUNS}")
log(f"Actual rows   : {len(df)}")
diff = len(df) - EXPECTED_RUNS
if diff == 0:
    log(f"[OK] All {EXPECTED_RUNS} runs present")
elif diff < 0:
    log(f"[FAIL] {-diff} runs MISSING -- check scenario_runner.py logs for failures")
else:
    log(f"[WARN] {diff} extra rows -- possible duplicates")

if "_source" in df.columns:
    src_counts = df["_source"].value_counts().to_dict()
    log(f"\nSource breakdown: {src_counts}")
    for src, expected in EXPECTED_SOURCES.items():
        actual = src_counts.get(src, 0)
        icon   = "[OK]" if actual == expected else "[FAIL]"
        log(f"  {icon} {src:12s}: {actual:3d}  (expected {expected})")
else:
    log("[WARN] _source column missing -- run scenario_runner.py v335+ to tag rows")

if "run_id" in df.columns:
    dupes = df["run_id"].duplicated().sum()
    log(f"\nDuplicate run_ids: {dupes}")
    if dupes:
        log(f"  Duplicate IDs: {df[df['run_id'].duplicated(keep=False)]['run_id'].tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCHEMA CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("3. SCHEMA CHECK")

present_params  = [c for c in PARAM_COLS  if c in df.columns]
missing_params  = [c for c in PARAM_COLS  if c not in df.columns]
present_targets = [c for c in ML_TARGETS  if c in df.columns]
missing_targets = [c for c in ML_TARGETS  if c not in df.columns]

log(f"Parameter cols  -- present: {len(present_params)}/{len(PARAM_COLS)}")
if missing_params:
    log(f"  MISSING: {missing_params}")
else:
    log("  [OK] All parameter columns present")

log(f"KPI target cols -- present: {len(present_targets)}/{len(ML_TARGETS)}")
if missing_targets:
    log(f"  MISSING: {missing_targets}")
else:
    log("  [OK] All KPI target columns present")


# ─────────────────────────────────────────────────────────────────────────────
# 4. NaN AUDIT
# ─────────────────────────────────────────────────────────────────────────────
section("4. NaN AUDIT")

all_cols   = present_params + present_targets
nan_counts = df[all_cols].isna().sum()
nan_cols   = nan_counts[nan_counts > 0]
if len(nan_cols) == 0:
    log("[OK] No NaN values in param or KPI columns")
else:
    log(f"NaN values found in {len(nan_cols)} columns:")
    for col, n in nan_cols.items():
        log(f"  {col:<40}  {n:3d} NaNs ({n/len(df)*100:.1f}%)")

# launch_time_0_60_s may be None if vehicle didn't reach 60 mph within timeout
if "launch_time_0_60_s" in df.columns:
    n_none = df["launch_time_0_60_s"].isna().sum()
    if n_none:
        log(f"\nlaunch_time_0_60_s: {n_none} None values "
            f"(vehicle did not reach 60 mph within timeout in {n_none} runs)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEAD CHANNELS
# ─────────────────────────────────────────────────────────────────────────────
section("5. DEAD CHANNELS (zero/near-zero variance KPIs)")

dead      = []
near_dead = []
for col in present_targets:
    std  = df[col].std()
    mean = df[col].mean()
    cv   = abs(std / mean) if mean != 0 else 0
    if std == 0:
        dead.append(col)
    elif cv < 0.01:
        near_dead.append((col, cv))

if dead:
    log(f"[FAIL] {len(dead)} DEAD channels (std=0):")
    for c in dead:
        log(f"    {c}  value={df[c].iloc[0]:.6f}")
else:
    log("[OK] No completely dead channels")

if near_dead:
    log(f"\n[WARN] {len(near_dead)} near-dead channels (CV < 1%):")
    for c, cv in near_dead:
        log(f"    {c}  CV={cv:.4f}")
else:
    log("[OK] No near-dead channels")


# ─────────────────────────────────────────────────────────────────────────────
# 6. OUTLIER FLAGGING  (IQR-based)
# ─────────────────────────────────────────────────────────────────────────────
section("6. OUTLIER FLAGGING (IQR x 3.0)")

df["_flag_outlier"] = False
outlier_counts = {}

for col in present_targets:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    mask   = (df[col] < lo) | (df[col] > hi)
    n_out  = mask.sum()
    if n_out:
        df.loc[mask, "_flag_outlier"] = True
        outlier_counts[col] = n_out

if outlier_counts:
    log(f"Outlier rows flagged per column:")
    for col, n in sorted(outlier_counts.items(), key=lambda x: -x[1]):
        log(f"  {col:<40}  {n:3d} rows")
else:
    log("[OK] No outliers detected at IQRx3.0")

total_flagged = df["_flag_outlier"].sum()
log(f"\nTotal flagged rows: {total_flagged} / {len(df)} "
    f"({total_flagged/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PARAMETER DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
section("7. PARAMETER DISTRIBUTIONS")

n_params = len(present_params)
ncols    = 4
nrows    = (n_params + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(present_params):
    ax   = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=30, edgecolor="white", linewidth=0.3, color="#4C8BF5")
    ax.axvline(data.mean(), color="red",    lw=1.5, linestyle="--", label=f"μ={data.mean():.3g}")
    ax.axvline(data.median(), color="green", lw=1.0, linestyle=":",  label=f"med={data.median():.3g}")
    ax.set_title(col, fontsize=7, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5)
    ax.set_xlabel(col, fontsize=6)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f"Parameter Distributions (all {len(df)} runs)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_param_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: 01_param_distributions.png")

# Summary stats
stats_df = df[present_params].describe().T[["mean", "std", "min", "max"]]
stats_df["CV%"] = (stats_df["std"] / stats_df["mean"].abs() * 100).round(1)
log(f"\nParameter summary stats:\n{stats_df.to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. KPI DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
section("8. KPI DISTRIBUTIONS")

n_tgts = len(present_targets)
ncols  = 3
nrows  = (n_tgts + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
axes = axes.flatten()

for i, col in enumerate(present_targets):
    ax   = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=30, edgecolor="white", linewidth=0.3, color="#F5A623")
    ax.axvline(data.mean(), color="red",  lw=1.5, label=f"μ={data.mean():.3f}")
    ax.axvline(data.median(), color="blue", lw=1.0, linestyle="--",
               label=f"med={data.median():.3f}")
    ax.set_title(col, fontsize=7, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("KPI Target Distributions", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_kpi_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: 02_kpi_distributions.png")

kpi_stats = df[present_targets].describe().T[["mean", "std", "min", "max"]]
kpi_stats["CV%"] = (kpi_stats["std"] / kpi_stats["mean"].abs() * 100).round(1)
log(f"\nKPI summary stats:\n{kpi_stats.to_string()}")

log("\nCoefficient of Variation -- higher = more signal for ML:")
for col in present_targets:
    cv = kpi_stats.loc[col, "CV%"] if col in kpi_stats.index else 0
    icon = "[OK]" if cv > 5 else "[WARN]"
    log(f"  {icon} {col:<40}  CV={cv:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PARAM -> KPI CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
section("9. PARAM -> KPI CORRELATIONS")

corr = pd.DataFrame(index=present_params, columns=present_targets, dtype=float)
for p in present_params:
    for t in present_targets:
        valid = df[[p, t]].dropna()
        if len(valid) > 5:
            r, _ = stats.pearsonr(valid[p], valid[t])
            corr.loc[p, t] = r

fig, ax = plt.subplots(
    figsize=(max(12, len(present_targets) * 0.9),
             max(6,  len(present_params)  * 0.5))
)
im = ax.imshow(corr.values.astype(float), cmap="RdBu_r",
               vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
ax.set_xticks(range(len(present_targets)))
ax.set_xticklabels(present_targets, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(present_params)))
ax.set_yticklabels(present_params, fontsize=8)
ax.set_title("Pearson r: Parameters -> KPI Targets", fontsize=11, fontweight="bold")
# Annotate cells
for i in range(len(present_params)):
    for j in range(len(present_targets)):
        val = corr.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color="white" if abs(val) > 0.5 else "black")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: 03_correlation_heatmap.png")

log("\nTop 15 strongest |r| correlations:")
flat = corr.stack().abs().dropna()
flat.index.names = ["param", "target"]
for (p, t), r in flat.sort_values(ascending=False).head(15).items():
    sign = "+" if corr.loc[p, t] > 0 else "-"
    log(f"  r={sign}{r:.3f}  {p:<22}  ->  {t}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. OAT SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("10. OAT SANITY CHECK")

if "_source" in df.columns and "_oat_param" in df.columns:
    oat_df  = df[df["_source"] == "oat"].copy()
    baseline = df[df["_source"] == "baseline"].iloc[0] if (df["_source"] == "baseline").any() else None
    log(f"OAT rows: {len(oat_df)}")

    # Pick representative targets to plot
    rep_targets = present_targets[:4]

    oat_params = [p for p in present_params if p in oat_df["_oat_param"].values]
    n_plot     = min(len(oat_params), 16)
    if n_plot == 0:
        log("  No OAT data to plot (OAT runs not yet collected)")
    else:
        ncols_oat  = 4
        nrows_oat  = (n_plot + ncols_oat - 1) // ncols_oat
        fig, axes  = plt.subplots(nrows_oat, ncols_oat, figsize=(20, nrows_oat * 4))
        axes       = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    if n_plot > 0:
        colours = plt.cm.tab10(np.linspace(0, 1, len(rep_targets)))

        for idx, param in enumerate(oat_params[:n_plot]):
            pdata = oat_df[oat_df["_oat_param"] == param].sort_values(param)
            ax    = axes[idx]
            for t, colour in zip(rep_targets, colours):
                if t in pdata.columns and pdata[t].notna().sum() > 0:
                    ax.plot(pdata[param], pdata[t], "o-", label=t, ms=5,
                            color=colour, lw=1.5)
                    if baseline is not None and t in baseline.index:
                        ax.axhline(baseline[t], color=colour, linestyle=":",
                                   lw=1, alpha=0.6)
            ax.set_title(param, fontsize=8, fontweight="bold")
            ax.set_xlabel(param, fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=5)

        for j in range(n_plot, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("OAT Sensitivity (dotted = baseline)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "04_oat_sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close()
        log("Saved: 04_oat_sensitivity.png")
else:
    log("Skipped -- _source/_oat_param columns not present in CSV")
    log("  (These are tagged automatically by scenario_runner.py build_configs)")


# ─────────────────────────────────────────────────────────────────────────────
# 11. LHS SPACE-FILL CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("11. LHS SPACE-FILL CHECK")

if "_source" in df.columns:
    lhs_df = df[df["_source"] == "lhs"][present_params]
    log(f"LHS runs: {len(lhs_df)}")
    log("Parameter coverage (% of design space sampled):")

    # Import design space ranges from settings
    try:
        from config.settings import PARAM_COLS
        # Try to get ranges from scenario_runner PARAM_RANGES
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "sr", ROOT / "data_collection" / "scenario_runner.py")
        if spec:
            sr = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sr)
            ranges = sr.PARAM_RANGES
        else:
            ranges = {}
    except Exception:
        ranges = {}

    for col in present_params:
        actual_lo = lhs_df[col].min()
        actual_hi = lhs_df[col].max()
        if col in ranges:
            lo, hi, _ = ranges[col]
            span     = hi - lo
            coverage = (actual_hi - actual_lo) / span * 100 if span > 0 else 0
            icon     = "[OK]" if coverage > 80 else "[WARN]"
            log(f"  {icon} {col:<22}: [{actual_lo:.4g}, {actual_hi:.4g}]  "
                f"coverage={coverage:.1f}%  design=[{lo}, {hi}]")
        else:
            log(f"  ? {col:<22}: [{actual_lo:.4g}, {actual_hi:.4g}]  "
                f"(design range unknown)")


# ─────────────────────────────────────────────────────────────────────────────
# 12. PAIRPLOT SAMPLE
# ─────────────────────────────────────────────────────────────────────────────
section("12. PAIRPLOT SAMPLE")

plot_params = present_params[:6]
plot_target = next((t for t in ["circle_max_lat_g", "slalom_max_lat_g"]
                    if t in df.columns), None)

if plot_target and len(plot_params) >= 2:
    n    = len(plot_params)
    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    vals = df[plot_target].values
    vmin, vmax = np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)
    cm   = plt.cm.plasma

    for i, pi in enumerate(plot_params):
        for j, pj in enumerate(plot_params):
            ax = axes[i][j]
            if i == j:
                ax.hist(df[pi].dropna(), bins=20,
                        color="#4C8BF5", edgecolor="white", lw=0.3)
                ax.set_xlabel(pi, fontsize=5)
            else:
                sc = ax.scatter(df[pj], df[pi], c=vals, cmap=cm,
                                vmin=vmin, vmax=vmax, s=5, alpha=0.7)
            ax.tick_params(labelsize=4)
            if j == 0:
                ax.set_ylabel(pi, fontsize=5)

    plt.colorbar(sc, ax=axes, label=plot_target, shrink=0.4)
    fig.suptitle(f"Pairplot -- colour = {plot_target}", fontsize=9)
    plt.savefig(PLOTS_DIR / "05_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: 05_pairplot.png  (colour = {plot_target})")


# ─────────────────────────────────────────────────────────────────────────────
# 13. DAMAGE / CRASH AUDIT
# ─────────────────────────────────────────────────────────────────────────────
section("13. DAMAGE / CRASH AUDIT")

# Check if any telemetry CSV recorded non-zero damage
damage_cols = [c for c in df.columns if "damage" in c.lower()]
if damage_cols:
    for col in damage_cols:
        n_damaged = (df[col] > 0).sum()
        log(f"  {col:<30}: {n_damaged} runs with damage > 0")
else:
    log("No damage columns found in sweep_results.csv")
    log("(Damage is recorded per-row in telemetry CSVs, not aggregated into sweep CSV)")


# ─────────────────────────────────────────────────────────────────────────────
# 14. EXPORT
# ─────────────────────────────────────────────────────────────────────────────
section("14. EXPORT")

n_flagged = int(df["_flag_outlier"].sum())
log(f"Total flagged rows  : {n_flagged} / {len(df)} ({n_flagged/len(df)*100:.1f}%)")
log(f"Clean rows          : {len(df) - n_flagged}")

df.to_csv(CSV_OUT, index=False)
log(f"Saved: {CSV_OUT}")
log("(All rows retained; _flag_outlier=True marks suspect rows)")


# ─────────────────────────────────────────────────────────────────────────────
# WRITE REPORT
# ─────────────────────────────────────────────────────────────────────────────
REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport  -> {REPORT_OUT}")
print(f"Plots   -> {PLOTS_DIR}")
print(f"Clean   -> {CSV_OUT}")
"""
data_processing/preprocess.py
==============================
Loads sweep_results.csv produced by scenario_runner.py, cleans it,
engineers features, and produces train/test splits for ML.

Column names match scenario_runner.py PARAM_RANGES exactly:
  Parameters : spring_F, spring_R, arb_spring_F, arb_spring_R,
               camber_FR, camber_RR, toe_FR, toe_RR,
               damp_bump_F, damp_bump_R, damp_rebound_F, damp_rebound_R,
               brakebias, brakestrength, lsdpreload_R, lsdlockcoef_R,
               lsdlockcoefrev_R, lsdpreload_F, lsdlockcoef_F,
               gear_1, gear_2, gear_3, gear_4, gear_5, gear_6,
               tyre_pressure_F, tyre_pressure_R
  KPI targets: launch_time_0_60_s, launch_peak_lon_g, launch_dist_3s_m,
               brake_stopping_distance_m, brake_peak_brake_g,
               circle_max_lat_g, circle_avg_lat_g, circle_speed_loss_ms,
               circle_understeer_proxy, slalom_max_lat_g,
               slalom_max_yaw_rate, slalom_yaw_rate_variance (log1p),
               slalom_avg_speed_ms,
               step_steer_peak_yaw_rate, step_steer_time_to_peak_s,
               step_steer_peak_lat_g, step_steer_yaw_overshoot,
               step_steer_settle_time_s
  Dropped KPIs: brake_yaw_rate_variance (always 0 on flat gridmap),
                circle_entry_speed_ms (CV=0.1%, constant across setups),
                highspeed_max_lat_g / highspeed_yaw_rate_var (CV<6% on flat straight,
                  ~35s per run for two near-dead channels -- replaced by step steer)
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    RESULTS_DIR, MODELS_DIR, SWEEP_CSV, CLEAN_CSV,
    TEST_SIZE, RANDOM_STATE, ML_TARGETS, PARAM_COLS,
)

log = logging.getLogger(__name__)

# ── Physical plausibility limits ──────────────────────────────────────────────
# Values outside these indicate BeamNG numerical instability at extreme
# parameter combinations rather than genuine vehicle behaviour.
PLAUSIBILITY_LIMITS = {
    "circle_max_lat_g":          2.5,
    "slalom_max_lat_g":          2.5,
    "brake_peak_brake_g":        2.0,
    "slalom_max_yaw_rate":       5.0,
    "brake_stopping_distance_m": 200.0,
}

# ── Failed-run sentinel filter ─────────────────────────────────────────────────
# When a BeamNG scenario crashes or the vehicle fails to complete a test the
# runner writes 0.0 for every KPI in that test block.  A brake_stopping_distance
# of exactly 0 m, a circle_max_lat_g of exactly 0 g, etc. are physically
# impossible, so any row containing such a zero is a failed simulation run that
# must be excluded.  We detect them via a launch-time ceiling: valid 0-60 times
# cluster tightly around ~7.7–8 s; anything above 15 s signals a run in which
# the vehicle never properly accelerated (all subsequent tests also invalid).
FAILED_RUN_LAUNCH_TIME_CEILING = 15.0   # seconds — hard cutoff for bad runs

ALWAYS_EXCLUDE = {
    "run_id", "config_name", "_source", "_oat_param", "_oat_level",
    "_flag_outlier",
}


def load_results(path: Path = None) -> pd.DataFrame:
    """Load sweep CSV. Always uses SWEEP_CSV (the new dataset) not the cached clean_data.csv."""
    if path is None:
        path = SWEEP_CSV
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    log.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path.name)
    return df


def clean_data(df: pd.DataFrame, use_flagged: bool = False) -> pd.DataFrame:
    """Remove invalid and physically implausible rows."""
    initial = len(df)

    # Drop rows flagged by validate_data.py
    if not use_flagged and "_flag_outlier" in df.columns:
        before = len(df)
        df = df[~df["_flag_outlier"].astype(bool)].copy()
        log.info("  Dropped %d flagged outlier rows", before - len(df))

    # Drop rows missing any ML target
    available_targets = [t for t in ML_TARGETS if t in df.columns]
    if not available_targets:
        raise ValueError(
            f"None of the ML_TARGETS found in dataframe. "
            f"Columns: {list(df.columns)[:10]}..."
        )
    df = df.dropna(subset=available_targets)

    # Drop None launch times (vehicle never reached 100 mph — test failed)
    if "launch_time_0_100_s" in df.columns:
        df = df[df["launch_time_0_100_s"].notna()]

    # Failed-run filter: drop rows where the vehicle never properly completed
    # the test sequence (launch_time >> normal → all KPIs zeroed by runner).
    if "launch_time_0_60_s" in df.columns:
        before = len(df)
        df = df[df["launch_time_0_60_s"] <= FAILED_RUN_LAUNCH_TIME_CEILING]
        removed_fr = before - len(df)
        if removed_fr:
            log.info(
                "  Failed-run filter: removed %d rows "
                "(launch_time_0_60_s > %.0f s → simulation failure)",
                removed_fr, FAILED_RUN_LAUNCH_TIME_CEILING,
            )

    # Physical plausibility filter
    before = len(df)
    for col, limit in PLAUSIBILITY_LIMITS.items():
        if col in df.columns:
            df = df[df[col] <= limit]
    removed = before - len(df)
    if removed:
        log.info("  Plausibility filter: removed %d physically implausible rows", removed)

    # Log1p-transform highly skewed variance targets so models see a more
    # Gaussian distribution.  Original columns are replaced in-place.
    for col in ("slalom_yaw_rate_variance",):
        if col in df.columns:
            df[col] = np.log1p(df[col])
            log.info("  log1p-transformed '%s'", col)

    log.info("Cleaning: %d → %d rows (%d removed)", initial, len(df), initial - len(df))
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that may help ML models learn physical interactions."""
    df = df.copy()

    def col(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    # Suspension balance — front/rear bias
    sf, sr = col("spring_F"), col("spring_R")
    df["feat_spring_ratio"]   = sf / (sr + 1e-6)
    df["feat_spring_total"]   = sf + sr
    df["feat_spring_balance"] = (sf - sr) / (sf + sr + 1e-6)

    af, ar = col("arb_spring_F"), col("arb_spring_R")
    df["feat_arb_ratio"]   = af / (ar + 1e-6)
    df["feat_arb_total"]   = af + ar
    df["feat_arb_balance"] = (af - ar) / (af + ar + 1e-6)

    # Overall platform stiffness
    df["feat_platform_stiffness"] = (sf + sr + af + ar) / 4.0

    # Tyre pressure
    pf, pr = col("tyre_pressure_F"), col("tyre_pressure_R")
    df["feat_tyre_balance"]  = pf - pr
    df["feat_tyre_total"]    = pf + pr

    # Geometry
    df["feat_camber_balance"] = col("camber_FR") - col("camber_RR")
    df["feat_toe_balance"]    = col("toe_FR") - col("toe_RR")

    # Brakes
    df["feat_brake_bias_dev"] = (col("brakebias") - 0.68).abs()

    # LSD
    lf = col("lsdlockcoef_F")
    lr = col("lsdlockcoef_R")
    df["feat_lsd_lock_balance"] = (lr - lf) / (lr + lf + 1e-6)
    df["feat_lsd_total_lock"]   = lf + lr
    df["feat_lsd_preload_total"] = col("lsdpreload_F") + col("lsdpreload_R")

    # Dampers
    dbf = col("damp_bump_F");    dbr = col("damp_bump_R")
    drf = col("damp_rebound_F"); drr = col("damp_rebound_R")
    df["feat_damp_bump_ratio"]      = dbf / (dbr + 1e-6)          # front/rear bump balance
    df["feat_damp_rebound_ratio"]   = drf / (drr + 1e-6)          # front/rear rebound balance
    df["feat_damp_bump_total"]      = dbf + dbr
    df["feat_damp_rebound_total"]   = drf + drr
    df["feat_damp_bumpreb_ratio_F"] = dbf / (drf + 1e-6)          # front compression vs extension
    df["feat_damp_bumpreb_ratio_R"] = dbr / (drr + 1e-6)          # rear compression vs extension

    # Gear ratios — spread and key ratios
    g1 = col("gear_1"); g6 = col("gear_6")
    df["feat_gear_spread"]     = g1 / (g6 + 1e-6)    # ratio of first to last gear
    df["feat_gear_1_6_ratio"]  = g1 * g6              # product captures overall ratio effect

    # Interaction: spring stiffness × tyre pressure (contact patch loading)
    df["feat_front_contact_load"] = sf * pf / 1e6
    df["feat_rear_contact_load"]  = sr * pr / 1e6

    # Critical damping ratios — ζ = c / (2 * sqrt(k * m_corner))
    # M_CORNER_KG is the approximate sprung mass per corner (vehicle ~1500 kg / 4).
    M_CORNER_KG = 375.0
    df["feat_crit_damp_ratio_bump_F"]    = dbf / (2.0 * np.sqrt(sf  * M_CORNER_KG + 1e-6))
    df["feat_crit_damp_ratio_bump_R"]    = dbr / (2.0 * np.sqrt(sr  * M_CORNER_KG + 1e-6))
    df["feat_crit_damp_ratio_rebound_F"] = drf / (2.0 * np.sqrt(sf  * M_CORNER_KG + 1e-6))
    df["feat_crit_damp_ratio_rebound_R"] = drr / (2.0 * np.sqrt(sr  * M_CORNER_KG + 1e-6))

    n_new = sum(1 for c in df.columns if c.startswith("feat_"))
    log.info("Feature engineering: added %d derived features", n_new)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all parameter + engineered feature columns, excluding targets."""
    param_cols = [c for c in PARAM_COLS if c in df.columns]
    feat_cols  = [c for c in df.columns if c.startswith("feat_")]
    exclude    = ALWAYS_EXCLUDE | set(ML_TARGETS)
    return [c for c in (param_cols + feat_cols) if c not in exclude]


def prepare_datasets(
        target: str,
        df: pd.DataFrame = None,
        use_flagged: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], RobustScaler]:
    """
    Full pipeline: load → clean → engineer → split → scale.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names                    : list of column names (for SHAP/importance)
    scaler                           : fitted RobustScaler (saved to models/saved/)
    """
    if df is None:
        df = load_results()

    df = clean_data(df, use_flagged=use_flagged)
    df = engineer_features(df)

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not in dataframe. "
            f"Available: {[c for c in df.columns if not c.startswith('feat_')]}"
        )

    feature_cols = get_feature_columns(df)
    needed       = feature_cols + [target]
    df_clean     = df[needed].dropna()

    if len(df_clean) < len(df):
        log.warning("  Dropped %d rows with NaN in features/target", len(df) - len(df_clean))

    if len(df_clean) < 20:
        raise ValueError(
            f"Only {len(df_clean)} clean rows for target '{target}' — too few to train"
        )

    X = df_clean[feature_cols].values.astype(float)
    y = df_clean[target].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler         = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / f"scaler_{target}.pkl")

    log.info(
        "  Dataset '%s': %d train / %d test / %d features",
        target, len(X_train), len(X_test), len(feature_cols),
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    df = load_results()
    df = clean_data(df)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    print(f"\nFinal dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}")

    print(f"\nML Targets:")
    for t in ML_TARGETS:
        if t in df.columns:
            valid = df[t].dropna()
            cv    = valid.std() / valid.mean() * 100 if valid.mean() != 0 else 0
            print(f"  {t:<40}  n={len(valid):3d}  "
                  f"mean={valid.mean():.4f}  CV={cv:.1f}%")
        else:
            print(f"  {t:<40}  MISSING")
"""
models/train_evaluate.py
=========================
Classical ML + ensemble models for each KPI target.

Models
------
1.  Random Forest
2.  Extra Trees
3.  XGBoost                (GPU)
4.  LightGBM               (GPU)
5.  CatBoost               (GPU)
6.  MLP Neural Network
7.  SVR
8.  Bayesian Ridge
9.  ElasticNet             (linear baseline)
10. Stacking Ensemble      (RF + XGB + LGBM → Ridge)

Install
-------
    pip install xgboost lightgbm catboost shap

Run
---
    python models/train_evaluate.py
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, StackingRegressor,
)
from sklearn.linear_model import ElasticNet, BayesianRidge, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MODELS_DIR, RESULTS_DIR, RANDOM_STATE, CV_FOLDS, ML_TARGETS
from data_processing.preprocess import prepare_datasets

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__)

try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False; log.warning("pip install xgboost")

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False; log.warning("pip install lightgbm")

try:
    import catboost as cb; HAS_CAT = True
except ImportError:
    HAS_CAT = False; log.warning("pip install catboost")

try:
    import shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False; log.warning("pip install shap")


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import subprocess
            return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
        except Exception:
            return False


def build_models(gpu: bool) -> Dict:
    models = {}

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
    )

    models["Extra Trees"] = ExtraTreesRegressor(
        n_estimators=200, max_features="sqrt", min_samples_split=5,
        n_jobs=-1, random_state=RANDOM_STATE,
    )

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            device="cuda" if gpu else "cpu",
            n_jobs=1 if gpu else -1,
            random_state=RANDOM_STATE, verbosity=0,
        )
    else:
        models["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.85, random_state=RANDOM_STATE,
        )

    if HAS_LGB:
        lgb_p = dict(n_estimators=300, learning_rate=0.05, max_depth=5,
                     num_leaves=31, subsample=0.85, colsample_bytree=0.85,
                     reg_alpha=0.1, reg_lambda=1.0,
                     n_jobs=1 if gpu else -1,
                     random_state=RANDOM_STATE, verbose=-1)
        if gpu:
            lgb_p["device"] = "gpu"
        models["LightGBM"] = lgb.LGBMRegressor(**lgb_p)

    if HAS_CAT:
        # R2 is not implemented on GPU in CatBoost — use RMSE as the eval
        # metric so the GPU is used end-to-end without a CPU metric fallback.
        models["CatBoost"] = cb.CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=5, l2_leaf_reg=3.0,
            task_type="GPU" if gpu else "CPU",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE, verbose=False,
        )

    models["MLP"] = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        solver="adam", learning_rate_init=1e-3, max_iter=500,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=RANDOM_STATE,
    )

    models["SVR"] = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
    )

    models["Bayesian Ridge"] = BayesianRidge(max_iter=500, tol=1e-4)

    models["ElasticNet"] = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)

    # Stacking ensemble
    # NOTE: base estimators must use CPU only — sklearn's StackingRegressor
    # runs internal CV with n_jobs=-1 (multiprocessing) which conflicts with
    # GPU context managers. XGBoost uses tree_method="hist" for fast CPU training.
    # Stacking base uses only sklearn-native estimators to avoid
    # is_regressor() validation failures with third-party wrappers (XGBoost).
    stack_base = [
        ("rf",  RandomForestRegressor(
            n_estimators=150, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_STATE)),
        ("gb",  GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05,
            max_depth=4, subsample=0.85,
            random_state=RANDOM_STATE)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300,
            early_stopping=True, random_state=RANDOM_STATE)),
    ]
    # cv=3: stacking fits len(base)*cv = 9 models internally — keep it lean
    models["Stacking Ensemble"] = StackingRegressor(
        estimators=stack_base, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1)

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test,
                   model_name: str, target: str) -> dict:
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    log.info("  %-22s  R2=%6.4f  RMSE=%8.5f  CV=%6.4f+/-%5.4f",
             model_name, r2, rmse, cv_scores.mean(), cv_scores.std())
    return {
        "model": model_name, "target": target,
        "cv_r2_mean": float(cv_scores.mean()), "cv_r2_std": float(cv_scores.std()),
        "test_r2": r2, "test_mse": float(mean_squared_error(y_test, y_pred)),
        "test_rmse": rmse, "test_mae": mae,
        "rel_rmse_%": rmse / (float(np.mean(np.abs(y_test))) + 1e-6) * 100,
    }


def plot_predictions(y_test, predictions: dict, target: str, save_dir: Path):
    n = len(predictions)
    cols = min(n, 4); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = np.array(axes).flatten()
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        r2 = r2_score(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.5, s=18, edgecolors="none", color="#2196F3")
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.5)
        ax.set_title(f"{name}\nR2={r2:.4f}", fontsize=8)
        ax.set_xlabel("Actual",fontsize=7); ax.set_ylabel("Predicted",fontsize=7)
        ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
    for ax in axes[n:]: ax.set_visible(False)
    fig.suptitle(f"Predictions — {target}", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_dir/f"predictions_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, target,
                            X_test, y_test, save_dir, top_n=20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    inner = model
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_"): inner = step; break
    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_
        idx = np.argsort(imp)[-top_n:]
        axes[0].barh([feature_names[i] for i in idx], imp[idx], color="#2196F3")
        axes[0].set_title(f"Native Importance\n{model_name}", fontsize=9)
    else:
        axes[0].text(0.5,0.5,"N/A",ha="center",va="center",transform=axes[0].transAxes)
    try:
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
        pidx = np.argsort(perm.importances_mean)[-top_n:]
        axes[1].barh([feature_names[i] for i in pidx], perm.importances_mean[pidx],
                     xerr=perm.importances_std[pidx], color="#4CAF50")
        axes[1].set_title(f"Permutation Importance\n{model_name}", fontsize=9)
    except Exception as e:
        log.warning("Permutation importance failed: %s", e)
    fig.suptitle(f"Feature Importance — {target}", fontsize=11)
    plt.tight_layout()
    safe = model_name.replace(" ","_").replace("(","").replace(")","")
    fig.savefig(save_dir/f"fi_{target}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap(model, X_train, feature_names, model_name, target, save_dir):
    if not HAS_SHAP: return
    try:
        inner = model
        if hasattr(model, "named_steps"):
            for v in model.named_steps.values():
                if hasattr(v, "predict"): inner = v; break
        if any(k in model_name for k in ("Forest","Tree","Boost","LGB","Cat")):
            explainer = shap.TreeExplainer(inner)
            sample    = X_train[:min(300,len(X_train))]
        else:
            sample    = X_train[:min(100,len(X_train))]
            explainer = shap.KernelExplainer(inner.predict, shap.sample(sample,50))
        sv = explainer.shap_values(sample)
        plt.figure(figsize=(10,7))
        shap.summary_plot(sv, sample, feature_names=feature_names, show=False, max_display=20)
        plt.title(f"SHAP — {model_name} -> {target}", fontsize=10)
        plt.tight_layout()
        safe = model_name.replace(" ","_").replace("(","").replace(")","")
        plt.savefig(save_dir/f"shap_{target}_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("    SHAP saved for %s", model_name)
    except Exception as e:
        log.warning("    SHAP failed for %s: %s", model_name, e)


def plot_learning_curve(model, X_train, y_train, model_name, target, save_dir):
    try:
        sizes, tr_sc, va_sc = learning_curve(
            model, X_train, y_train, cv=5, scoring="r2",
            train_sizes=np.linspace(0.2,1.0,6), n_jobs=1)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sizes, tr_sc.mean(1),"o-",label="Train R2",color="#2196F3")
        ax.fill_between(sizes,tr_sc.mean(1)-tr_sc.std(1),tr_sc.mean(1)+tr_sc.std(1),
                        alpha=0.2,color="#2196F3")
        ax.plot(sizes, va_sc.mean(1),"o-",label="Val R2",color="#F44336")
        ax.fill_between(sizes,va_sc.mean(1)-va_sc.std(1),va_sc.mean(1)+va_sc.std(1),
                        alpha=0.2,color="#F44336")
        ax.set_xlabel("Training samples"); ax.set_ylabel("R2")
        ax.set_title(f"Learning Curve — {model_name} -> {target}")
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        safe = model_name.replace(" ","_").replace("(","").replace(")","")
        fig.savefig(save_dir/f"lc_{target}_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log.warning("Learning curve failed for %s: %s", model_name, e)


def plot_model_comparison(results_df: pd.DataFrame, save_dir: Path):
    targets = sorted(results_df["target"].unique())
    models  = results_df["model"].unique()
    colours = dict(zip(models, plt.cm.tab20(np.linspace(0,1,len(models)))))
    ncols = 3; nrows = (len(targets)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.array(axes).flatten()
    for ax, tgt in zip(axes, targets):
        sub  = results_df[results_df["target"]==tgt].sort_values("test_r2")
        bars = ax.barh(sub["model"], sub["test_r2"],
                       color=[colours[m] for m in sub["model"]], edgecolor="white", lw=0.5)
        ax.set_title(tgt, fontsize=9, fontweight="bold")
        ax.set_xlabel("Test R2", fontsize=8)
        ax.axvline(0,color="black",lw=0.5)
        ax.axvline(0.85,color="green",lw=0.8,ls="--",alpha=0.5)
        for bar,val in zip(bars,sub["test_r2"]):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axes[len(targets):]: ax.set_visible(False)
    fig.suptitle("Classical ML — Test R2 by Target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir/"classical_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Comparison chart saved")


def run_full_evaluation():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    gpu = _has_gpu()
    log.info("GPU: %s | XGB:%s LGB:%s CAT:%s",
             gpu, HAS_XGB, HAS_LGB, HAS_CAT)

    all_results = []
    best_models = {}

    for target in ML_TARGETS:
        log.info("\n%s\nTARGET: %s\n%s","="*60,target,"="*60)
        try:
            X_train,X_test,y_train,y_test,feat_names,scaler = prepare_datasets(target)
        except Exception as e:
            log.error("Dataset prep failed for %s: %s", target, e); continue

        models  = build_models(gpu)
        fig_dir = FIGURES_DIR / target.replace(" ","_")
        fig_dir.mkdir(parents=True, exist_ok=True)

        best_r2 = -np.inf; best_name = None; best_model = None; preds = {}
        for name, model in models.items():
            log.info("\n  Training: %s", name)
            try:
                res = evaluate_model(model,X_train,X_test,y_train,y_test,name,target)
                all_results.append(res)
                preds[name] = model.predict(X_test)
                if res["test_r2"] > best_r2:
                    best_r2=res["test_r2"]; best_name=name; best_model=model
            except Exception as e:
                log.error("  %s FAILED: %s", name, e, exc_info=True)

        if best_model is None: continue
        log.info("\n  Best for '%s': %s (R2=%.4f)", target, best_name, best_r2)
        best_models[target] = (best_name, best_model, feat_names, scaler)
        joblib.dump(best_model,  MODELS_DIR/f"best_{target}.pkl")
        joblib.dump(feat_names, MODELS_DIR/f"features_{target}.pkl")

        plot_predictions(y_test, preds, target, fig_dir)
        plot_feature_importance(best_model,feat_names,best_name,target,X_test,y_test,fig_dir)
        plot_shap(best_model,X_train,feat_names,best_name,target,fig_dir)
        # Learning curve only for best model to avoid excessive compute
        if best_name not in ("Stacking Ensemble", "SVR"):
            plot_learning_curve(best_model,X_train,y_train,best_name,target,fig_dir)

    results_df = pd.DataFrame(all_results)
    out = RESULTS_DIR/"model_evaluation_results.csv"
    results_df.to_csv(out, index=False)
    log.info("\nResults -> %s", out)
    plot_model_comparison(results_df, FIGURES_DIR)
    summary = results_df.pivot_table(
        values="test_r2",index="model",columns="target",aggfunc="first").round(4)
    log.info("\nTest R2 Summary:\n%s", summary.to_string())
    return best_models, results_df

if __name__ == "__main__":
    run_full_evaluation()
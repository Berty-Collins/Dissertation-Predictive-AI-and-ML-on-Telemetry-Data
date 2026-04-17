"""
models/train_ai_models.py
==========================
Foundation model and advanced AI regressors for each KPI target.

Models
------
1.  TabPFN v2              — tabular foundation model, in-context learning
2.  TabNet                 — attention-based tabular deep learning
3.  PatchTST               — patch Transformer, fine-tuned on tabular data
4.  Chronos-2              — time-series FM adapted for tabular regression
5.  AutoGluon              — automated ML with neural net + boosting ensemble

Install
-------
    pip install tabpfn
    pip install pytorch-tabnet
    pip install transformers torch accelerate
    pip install autogluon.tabular
    pip install chronos-forecasting

HuggingFace models (add to your account):
    amazon/chronos-t5-small     -- Chronos weights (needed for run_chronos)

Run
---
    python models/train_ai_models.py
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MODELS_DIR, RESULTS_DIR, RANDOM_STATE, CV_FOLDS, ML_TARGETS
from data_processing.preprocess import prepare_datasets

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    from tabpfn import TabPFNRegressor; HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False; log.warning("pip install tabpfn")

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch as _torch_tabnet; HAS_TABNET = True
except ImportError:
    HAS_TABNET = False; log.warning("pip install pytorch-tabnet")

try:
    import torch
    from transformers import (
        PatchTSTConfig, PatchTSTForRegression,
        Trainer, TrainingArguments,
    )
    from torch.utils.data import Dataset as TorchDataset
    HAS_PATCHTST = True
except ImportError:
    HAS_PATCHTST = False; log.warning("pip install transformers torch")

try:
    from chronos import BaseChronosPipeline; HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False; log.warning("pip install chronos-forecasting")

try:
    from autogluon.tabular import TabularPredictor; HAS_AG = True
except ImportError:
    HAS_AG = False; log.warning("pip install autogluon.tabular")


def _has_gpu() -> bool:
    try:
        import torch; return torch.cuda.is_available()
    except ImportError:
        try:
            import subprocess
            return subprocess.run(["nvidia-smi"],capture_output=True).returncode==0
        except Exception: return False


def _compute_metrics(model_name, target, y_test, y_pred, cv_scores) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    log.info("  %-32s  R2=%6.4f  RMSE=%8.5f  CV=%6.4f+/-%5.4f",
             model_name, r2, rmse, float(cv_scores.mean()), float(cv_scores.std()))
    return {
        "model": model_name, "target": target,
        "cv_r2_mean": float(cv_scores.mean()), "cv_r2_std": float(cv_scores.std()),
        "test_r2": r2, "test_mse": float(mean_squared_error(y_test,y_pred)),
        "test_rmse": rmse, "test_mae": mae,
        "rel_rmse_%": rmse/(float(np.mean(np.abs(y_test)))+1e-6)*100,
    }


def _plot_predictions(y_test, predictions: dict, target: str, save_dir: Path):
    n = len(predictions)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        r2 = r2_score(y_test, y_pred)
        ax.scatter(y_test,y_pred,alpha=0.55,s=22,edgecolors="none",color="#2196F3")
        lo,hi = min(y_test.min(),y_pred.min()),max(y_test.max(),y_pred.max())
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.5,label="Perfect fit")
        ax.set_xlabel(f"Actual {target}"); ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{name}\nR2={r2:.4f}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f"AI Models -- {target}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(save_dir/f"ai_predictions_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# 1. TabPFN v2
# =============================================================================
def run_tabpfn(X_train, X_test, y_train, y_test, target) -> Tuple[dict, np.ndarray]:
    log.info("  [TabPFN-v2] 5-fold CV...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for fold,(ti,vi) in enumerate(kf.split(X_train)):
        m = TabPFNRegressor(n_estimators=8, random_state=RANDOM_STATE)
        m.fit(X_train[ti], y_train[ti])
        cv_scores.append(r2_score(y_train[vi], m.predict(X_train[vi])))
        log.info("    fold %d/%d  R2=%.4f", fold+1, CV_FOLDS, cv_scores[-1])
    model = TabPFNRegressor(n_estimators=8, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, MODELS_DIR/f"tabpfn_{target}.pkl")
    return _compute_metrics("TabPFN-v2", target, y_test, y_pred,
                            np.array(cv_scores)), y_pred


# =============================================================================
# 2. TabNet
# =============================================================================
def run_tabnet(X_train, X_test, y_train, y_test, target) -> Tuple[dict, np.ndarray]:
    import torch
    gpu = _has_gpu()
    log.info("  [TabNet] 5-fold CV...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for fold,(ti,vi) in enumerate(kf.split(X_train)):
        m = TabNetRegressor(
            n_d=32, n_a=32, n_steps=5, gamma=1.5,
            n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 2e-3, "weight_decay": 1e-5},
            scheduler_params={"mode": "min", "patience": 5, "factor": 0.5},
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            mask_type="entmax",
            device_name="cuda" if gpu else "cpu",
            verbose=0,
            seed=RANDOM_STATE,
        )
        m.fit(
            X_train[ti], y_train[ti].reshape(-1,1),
            eval_set=[(X_train[vi], y_train[vi].reshape(-1,1))],
            max_epochs=200, patience=20,
            batch_size=min(256,len(ti)),
            virtual_batch_size=min(128,len(ti)//2),
        )
        pv = m.predict(X_train[vi]).flatten()
        cv_scores.append(r2_score(y_train[vi], pv))
        log.info("    fold %d/%d  R2=%.4f", fold+1, CV_FOLDS, cv_scores[-1])

    log.info("  [TabNet] final fit...")
    model = TabNetRegressor(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-3, "weight_decay": 1e-5},
        scheduler_params={"mode": "min", "patience": 5, "factor": 0.5},
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type="entmax",
        device_name="cuda" if gpu else "cpu",
        verbose=0,
        seed=RANDOM_STATE,
    )
    model.fit(
        X_train, y_train.reshape(-1,1),
        eval_set=[(X_test, y_test.reshape(-1,1))],
        max_epochs=300, patience=30,
        batch_size=min(256,len(X_train)),
        virtual_batch_size=min(128,len(X_train)//2),
    )
    y_pred = model.predict(X_test).flatten()
    save_path = str(MODELS_DIR/f"tabnet_{target}")
    model.save_model(save_path)
    return _compute_metrics("TabNet", target, y_test, y_pred,
                            np.array(cv_scores)), y_pred


# =============================================================================
# 3. PatchTST (fine-tuned)
# =============================================================================
class _FeatureDataset(TorchDataset if HAS_PATCHTST else object):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        item = {"past_values": self.X[idx].unsqueeze(-1)}
        if self.y is not None: item["target_values"] = self.y[idx].unsqueeze(0)
        return item


def _build_patchtst(seq_len):
    from transformers import PatchTSTConfig, PatchTSTForRegression
    cfg = PatchTSTConfig(
        num_input_channels=1, context_length=seq_len,
        patch_length=4, patch_stride=4, prediction_length=1,
        num_targets=1, d_model=64, num_attention_heads=4,
        num_hidden_layers=3, ffn_dim=128, dropout=0.15,
        head_dropout=0.1, pooling_type="mean", channel_attention=False,
    )
    return PatchTSTForRegression(cfg)


def _train_one_patchtst(X_tr, y_tr, X_va, y_va, seq_len, epochs, output_dir):
    gpu = _has_gpu()
    model = _build_patchtst(seq_len)
    args  = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=min(32,len(X_tr)),
        learning_rate=1e-3, weight_decay=1e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=999, report_to="none",
        seed=RANDOM_STATE, fp16=gpu, no_cuda=not gpu,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=_FeatureDataset(X_tr,y_tr),
                      eval_dataset=_FeatureDataset(X_va,y_va))
    trainer.train()
    raw = trainer.predict(_FeatureDataset(X_va,y_va)).predictions
    if isinstance(raw,tuple):
        raw = next(r for r in raw if hasattr(r,"__len__") and len(r)==len(X_va))
    return model, np.array(raw,dtype=float).reshape(-1)


def run_patchtst(X_train, X_test, y_train, y_test, target) -> Tuple[dict, np.ndarray]:
    seq_len = X_train.shape[1]
    log.info("  [PatchTST] 5-fold CV (20 epochs per fold)...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for fold,(ti,vi) in enumerate(kf.split(X_train)):
        _,preds = _train_one_patchtst(
            X_train[ti],y_train[ti],X_train[vi],y_train[vi],
            seq_len, epochs=20, output_dir=f"/tmp/ptst_{target}_f{fold}")
        cv_scores.append(r2_score(y_train[vi],preds))
        log.info("    fold %d/%d  R2=%.4f",fold+1,CV_FOLDS,cv_scores[-1])
    log.info("  [PatchTST] final fit (80 epochs)...")
    save_path = str(MODELS_DIR/f"patchtst_{target}")
    model,_ = _train_one_patchtst(
        X_train,y_train,X_test,y_test,seq_len,epochs=80,output_dir=save_path)
    gpu = _has_gpu()
    raw = Trainer(model=model,
                  args=TrainingArguments(output_dir="/tmp/ptst_eval",
                                         report_to="none",no_cuda=not gpu)
                  ).predict(_FeatureDataset(X_test,y_test)).predictions
    if isinstance(raw,tuple):
        raw = next(r for r in raw if hasattr(r,"__len__") and len(r)==len(X_test))
    y_pred = np.array(raw,dtype=float).reshape(-1)
    model.save_pretrained(save_path)
    return _compute_metrics("PatchTST",target,y_test,y_pred,
                            np.array(cv_scores)), y_pred


# =============================================================================
# 4. Chronos-2 (zero-shot via nearest-neighbour context)
# =============================================================================
class _ChronosAdapter:
    def __init__(self, k=20):
        self.k=k; self.nn_=None; self._y=None; self._pipe=None
    def _load(self):
        if self._pipe is None:
            log.info("    Loading Chronos-2 weights (may download ~500 MB)...")
            import torch
            self._pipe = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cuda" if _has_gpu() else "cpu",
                dtype=torch.float32,
            )
    def fit(self, X, y):
        self._load()
        self.nn_ = NearestNeighbors(n_neighbors=self.k,metric="euclidean").fit(X)
        self._y  = y.copy(); return self
    def predict(self, X):
        import torch
        _,idx = self.nn_.kneighbors(X); preds=[]
        for row in idx:
            ctx = torch.tensor(self._y[row[::-1]].astype(np.float32)).unsqueeze(0)
            fc  = self._pipe.predict(ctx, prediction_length=1, num_samples=20)
            preds.append(float(np.median(fc[0,:,0].numpy())))
        return np.array(preds,dtype=float)
    def cv_scores(self, X, y, folds=5):
        kf=KFold(n_splits=folds,shuffle=True,random_state=RANDOM_STATE); sc=[]
        for fold,(ti,vi) in enumerate(kf.split(X)):
            self.fit(X[ti],y[ti])
            sc.append(r2_score(y[vi],self.predict(X[vi])))
            log.info("    fold %d/%d  R2=%.4f",fold+1,folds,sc[-1])
        return sc


def run_chronos(X_train, X_test, y_train, y_test, target) -> Tuple[dict, np.ndarray]:
    adapter = _ChronosAdapter(k=20)
    log.info("  [Chronos-2] 5-fold CV...")
    cv = adapter.cv_scores(X_train, y_train)
    log.info("  [Chronos-2] predicting test set...")
    adapter.fit(X_train, y_train)
    y_pred = adapter.predict(X_test)
    return _compute_metrics("Chronos-2 (zero-shot)",target,y_test,y_pred,
                            np.array(cv)), y_pred


# =============================================================================
# 5. AutoGluon
# =============================================================================
def run_autogluon(X_train, X_test, y_train, y_test,
                  feature_names, target) -> Tuple[dict, np.ndarray]:
    import tempfile, os
    log.info("  [AutoGluon] fitting (time_limit=300s)...")
    gpu = _has_gpu()

    # AutoGluon expects DataFrames
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train["__target__"] = y_train
    df_test  = pd.DataFrame(X_test,  columns=feature_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        predictor = TabularPredictor(
            label="__target__",
            path=tmpdir,
            problem_type="regression",
            eval_metric="r2",
            verbosity=1,
        ).fit(
            df_train,
            time_limit=300,
            presets="best_quality",
            num_gpus=1 if gpu else 0,
        )
        y_pred    = predictor.predict(df_test).values
        lb        = predictor.leaderboard(df_train, silent=True)
        log.info("  [AutoGluon] leaderboard:\n%s", lb[["model","score_val"]].to_string())

        # 5-fold CV on training set using best model
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        for fold,(ti,vi) in enumerate(kf.split(X_train)):
            dtr = pd.DataFrame(X_train[ti],columns=feature_names)
            dtr["__target__"] = y_train[ti]
            dva = pd.DataFrame(X_train[vi],columns=feature_names)
            p_cv = TabularPredictor(
                label="__target__", path=os.path.join(tmpdir,f"cv{fold}"),
                problem_type="regression", eval_metric="r2", verbosity=0,
            ).fit(dtr, time_limit=60, presets="medium_quality", num_gpus=1 if gpu else 0)
            cv_scores.append(r2_score(y_train[vi], p_cv.predict(dva).values))
            log.info("    fold %d/%d  R2=%.4f",fold+1,CV_FOLDS,cv_scores[-1])

    return _compute_metrics("AutoGluon",target,y_test,y_pred,
                            np.array(cv_scores)), y_pred


# =============================================================================
# REGISTRY & MAIN
# =============================================================================
AI_MODELS = [
    ("TabPFN-v2",             HAS_TABPFN,   lambda *a: run_tabpfn(*a)),
    ("TabNet",                HAS_TABNET,   lambda *a: run_tabnet(*a)),
    ("PatchTST",              HAS_PATCHTST, lambda *a: run_patchtst(*a)),
    ("Chronos-2 (zero-shot)", HAS_CHRONOS,  lambda *a: run_chronos(*a)),
    ("AutoGluon",             HAS_AG,       None),  # special signature
]


def _save_results(all_results):
    df      = pd.DataFrame(all_results)
    ai_path = RESULTS_DIR/"model_evaluation_results_ai.csv"
    df.to_csv(ai_path, index=False)
    log.info("AI results -> %s", ai_path)
    classic = RESULTS_DIR/"model_evaluation_results.csv"
    if classic.exists():
        combined = pd.concat([pd.read_csv(classic), df], ignore_index=True)
        combined.to_csv(RESULTS_DIR/"model_evaluation_results_all.csv", index=False)
        log.info("Combined -> model_evaluation_results_all.csv")
    summary = df.pivot_table(values="test_r2",index="model",
                             columns="target",aggfunc="first").round(4)
    log.info("\nTest R2 summary:\n%s", summary.to_string())
    return df


def _plot_combined(results_df):
    classic = RESULTS_DIR/"model_evaluation_results.csv"
    if classic.exists():
        results_df = pd.concat([pd.read_csv(classic),results_df],ignore_index=True)
    targets = sorted(results_df["target"].unique())
    models  = results_df["model"].unique()
    colours = dict(zip(models,plt.cm.tab20(np.linspace(0,1,len(models)))))
    ncols=4; nrows=(len(targets)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4*nrows),sharey=False)
    axes=np.array(axes).flatten()
    for ax,tgt in zip(axes,targets):
        sub=results_df[results_df["target"]==tgt].sort_values("test_r2")
        bars=ax.barh(sub["model"],sub["test_r2"],
                     color=[colours[m] for m in sub["model"]],edgecolor="white",lw=0.5)
        ax.set_title(tgt,fontsize=9,fontweight="bold")
        ax.set_xlabel("Test R2",fontsize=8)
        ax.axvline(0,color="black",lw=0.5)
        ax.axvline(0.85,color="green",lw=0.8,ls="--",alpha=0.5)
        for bar,val in zip(bars,sub["test_r2"]):
            ax.text(bar.get_width()+0.005,bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}",va="center",fontsize=6)
        ax.tick_params(labelsize=7)
    for ax in axes[len(targets):]: ax.set_visible(False)
    fig.suptitle("All Models -- Test R2 by Target",fontsize=14,fontweight="bold")
    plt.tight_layout()
    out=FIGURES_DIR/"combined_model_comparison.png"
    fig.savefig(out,dpi=150,bbox_inches="tight")
    plt.close()
    log.info("Combined comparison -> %s",out)


def run_ai_evaluation():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    gpu = _has_gpu()
    log.info("GPU: %s", gpu)

    available = [(n,fn) for n,avail,fn in AI_MODELS if avail and fn is not None]
    has_ag    = HAS_AG
    skipped   = [n for n,avail,_ in AI_MODELS if not avail]
    if skipped: log.warning("Skipping (not installed): %s", skipped)

    all_results = []

    for target in ML_TARGETS:
        log.info("\n%s\nTARGET: %s\n%s","="*60,target,"="*60)
        try:
            X_train,X_test,y_train,y_test,feat_names,_ = prepare_datasets(target)
        except Exception as e:
            log.error("Dataset prep failed for %s: %s",target,e); continue

        fig_dir = FIGURES_DIR/target
        fig_dir.mkdir(parents=True,exist_ok=True)
        preds   = {}

        for model_name, eval_fn in available:
            log.info("\n  --- %s ---", model_name)
            try:
                res, y_pred = eval_fn(X_train,X_test,y_train,y_test,target)
                all_results.append(res); preds[model_name]=y_pred
            except Exception as e:
                log.error("  %s FAILED: %s",model_name,e,exc_info=True)

        # AutoGluon (needs feature_names)
        if has_ag:
            log.info("\n  --- AutoGluon ---")
            try:
                res,y_pred = run_autogluon(
                    X_train,X_test,y_train,y_test,feat_names,target)
                all_results.append(res); preds["AutoGluon"]=y_pred
            except Exception as e:
                log.error("  AutoGluon FAILED: %s",e,exc_info=True)

        if preds:
            _plot_predictions(y_test,preds,target,fig_dir)

    if not all_results:
        log.error("No results. Check dependencies."); return
    df = _save_results(all_results)
    _plot_combined(df)
    return df

if __name__ == "__main__":
    run_ai_evaluation()
"""
models/_ag_worker.py
====================
AutoGluon subprocess worker — called by run_autogluon() in train_ai_models.py.

WHY A SUBPROCESS?
-----------------
On Windows, AutoGluon's worker threads hold OS-level file locks on model
artefacts even after predictor.fit() returns.  This prevents shutil.rmtree
from cleaning up the temporary directory.  The leftover fitted model is then
detected by the *next* target's TabularPredictor, which raises:
    AssertionError: Learner is already fit.

Running each target in its own subprocess guarantees:
  1. Complete isolation — no shared AutoGluon module-level globals.
  2. All file locks are released on subprocess exit (OS process termination).
  3. The parent can successfully shutil.rmtree the tmpdir after the child exits.

Usage (internal — do not call directly):
    python _ag_worker.py <input.pkl> <output.pkl>
"""

import pickle
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: _ag_worker.py <input.pkl> <output.pkl>", file=sys.stderr)
        sys.exit(2)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, "rb") as fh:
        d = pickle.load(fh)

    X_train       = d["X_train"]
    X_test        = d["X_test"]
    y_train       = d["y_train"]
    feature_names = d["feature_names"]
    target        = d["target"]
    gpu           = d["gpu"]
    cv_folds      = d["cv_folds"]
    random_state  = d["random_state"]

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("autogluon.tabular not installed — install with: pip install autogluon.tabular",
              file=sys.stderr)
        sys.exit(3)

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train["__target__"] = y_train
    df_test  = pd.DataFrame(X_test,  columns=feature_names)

    # ── Main fit ──────────────────────────────────────────────────────────────
    # eval_metric="rmse" avoids "R2 is not implemented on GPU" from CatBoost.
    # Our caller computes the actual R² from y_pred vs y_test after the fact.
    tmpdir = tempfile.mkdtemp(prefix=f"ag_{target[:12]}_")
    predictor = TabularPredictor(
        label="__target__",
        path=tmpdir,
        problem_type="regression",
        eval_metric="rmse",
        verbosity=1,
    ).fit(
        df_train,
        time_limit=300,
        presets="best_quality",
        num_gpus=1 if gpu else 0,
        excluded_model_types=["FASTAI"],
    )
    y_pred = predictor.predict(df_test).to_numpy().tolist()

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = []
    cv_dirs   = []
    for fold, (ti, vi) in enumerate(kf.split(X_train)):
        dtr = pd.DataFrame(X_train[ti], columns=feature_names)
        dtr["__target__"] = y_train[ti]
        dva = pd.DataFrame(X_train[vi], columns=feature_names)

        cv_path = tempfile.mkdtemp(prefix=f"ag_{target[:8]}_cv{fold}_")
        cv_dirs.append(cv_path)

        p_cv = TabularPredictor(
            label="__target__",
            path=cv_path,
            problem_type="regression",
            eval_metric="rmse",
            verbosity=0,
        ).fit(
            dtr,
            time_limit=60,
            presets="medium_quality",
            num_gpus=1 if gpu else 0,
            excluded_model_types=["FASTAI"],
        )
        score = float(r2_score(y_train[vi], p_cv.predict(dva).to_numpy()))
        cv_scores.append(score)
        print(f"    [AutoGluon CV] fold {fold + 1}/{cv_folds}  R2={score:.4f}", flush=True)

    # ── Write output ──────────────────────────────────────────────────────────
    # tmpdir and cv_dirs are returned to the parent so it can clean them up
    # AFTER this subprocess exits (i.e. after all file locks are released).
    with open(output_path, "wb") as fh:
        pickle.dump({
            "y_pred":    y_pred,
            "cv_scores": cv_scores,
            "tmpdir":    tmpdir,
            "cv_dirs":   cv_dirs,
        }, fh)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

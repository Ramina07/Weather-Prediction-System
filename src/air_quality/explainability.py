"""Experiment E6: model interpretability (associational, not causal).

* Built-in impurity/gain importances for Random Forest and XGBoost.
* Standardized coefficients for Ridge Regression.
* Permutation importance (MAE increase) on the held-out 2025 test set for
  every frozen benchmark model.
* Optional SHAP summary for the selected tree-based model (if ``shap`` is
  installed).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import Config, get_logger
from .evaluation import UNIT, _label, _plt, _save
from .training import TARGET, load_model_frame


def _transform_until_model(pipe, X: pd.DataFrame) -> np.ndarray:
    Z = X
    for name, step in pipe.steps[:-1]:
        Z = step.transform(Z)
    return np.asarray(Z, dtype=float)


def run_explainability(cfg: Config, n_repeats: int = 10) -> dict:
    log = get_logger()
    bdir = cfg.path("models_dir") / "benchmark"
    mdir, fdir = cfg.path("metrics_dir"), cfg.path("figures_dir")
    df = load_model_frame(cfg)
    te = df[df["split"] == "test"]
    sel = json.loads((mdir / "model_selection.json").read_text(encoding="utf-8"))
    best = sel["best"]["model"]
    out: dict = {}
    perm_rows, intrinsic_rows = [], []
    for name in ["ridge", "random_forest", "xgboost", "svr", "ensemble"]:
        path = bdir / f"{name}.joblib"
        if not path.exists():
            continue
        b = joblib.load(path)
        pipe, feats, used = b["pipeline"], b["features"], b["selected_features"]
        est = pipe.named_steps["model"] if hasattr(pipe, "named_steps") else None
        if hasattr(est, "feature_importances_"):
            for f, v in zip(used, est.feature_importances_):
                intrinsic_rows.append({"model": name, "feature": f, "kind": "impurity/gain importance", "value": float(v)})
        if hasattr(est, "coef_"):
            for f, v in zip(used, np.ravel(est.coef_)):
                intrinsic_rows.append({"model": name, "feature": f, "kind": "standardized coefficient", "value": float(v)})
        pi = permutation_importance(pipe, te[feats], te[TARGET].to_numpy(), scoring="neg_mean_absolute_error",
                                    n_repeats=n_repeats, random_state=cfg.seed, n_jobs=1)
        for f, m, s in zip(feats, pi.importances_mean, pi.importances_std):
            perm_rows.append({"model": name, "feature": f, "mae_increase_mean": float(m), "mae_increase_std": float(s)})
        log.info("Permutation importance computed for %s", name)
    perm = pd.DataFrame(perm_rows)
    intr = pd.DataFrame(intrinsic_rows)
    perm.to_csv(mdir / "E6_permutation_importance_test.csv", index=False)
    intr.to_csv(mdir / "E6_intrinsic_importance.csv", index=False)

    plt = _plt()
    models_present = [m for m in ["ridge", "random_forest", "xgboost", "svr", "ensemble"] if m in set(perm["model"])]
    fig, axes = plt.subplots(1, len(models_present), figsize=(4.2 * len(models_present), 5), squeeze=False)
    for ax, m in zip(axes[0], models_present):
        g = perm[perm.model == m].nlargest(12, "mae_increase_mean").iloc[::-1]
        ax.barh(g["feature"], g["mae_increase_mean"], xerr=g["mae_increase_std"], color="#4C72B0", alpha=0.85)
        ax.set_title(_label(m) + (" (selected)" if m == best else ""))
        ax.set_xlabel(f"MAE increase when permuted ({UNIT})")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Fig. 12. Permutation importance on the 2025 test set (top 12 features per model)")
    _save(fig, fdir / "12_feature_importance_permutation.png")

    trees = intr[intr.kind == "impurity/gain importance"]
    if len(trees):
        ms = list(dict.fromkeys(trees["model"]))
        fig, axes = plt.subplots(1, len(ms), figsize=(5 * len(ms), 5), squeeze=False)
        for ax, m in zip(axes[0], ms):
            g = trees[trees.model == m].nlargest(12, "value").iloc[::-1]
            ax.barh(g["feature"], g["value"], color="#55A868" if m == "random_forest" else "#C44E52")
            ax.set_title(f"{_label(m)}: built-in importance")
            ax.set_xlabel("Relative importance (sums to 1)")
            ax.tick_params(axis="y", labelsize=8)
        fig.suptitle("Fig. 12b. Tree-model intrinsic feature importance")
        _save(fig, fdir / "12b_feature_importance_trees.png")
    coefs = intr[intr.kind == "standardized coefficient"]
    if len(coefs):
        g = coefs.reindex(coefs["value"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.barh(g["feature"], g["value"], color=np.where(g["value"] > 0, "#C44E52", "#4C72B0"))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Coefficient ({UNIT} per 1 SD of feature)")
        ax.set_title("Fig. 12c. Ridge Regression standardized coefficients (top 15 |coef|)")
        _save(fig, fdir / "12c_ridge_coefficients.png")

    # optional SHAP
    tree_best = best if best in ("random_forest", "xgboost") else None
    if tree_best is None:
        cand = [m for m in ("xgboost", "random_forest") if (bdir / f"{m}.joblib").exists()]
        tree_best = cand[0] if cand else None
    out["shap"] = "skipped"
    if tree_best:
        try:
            import shap  # type: ignore
            b = joblib.load(bdir / f"{tree_best}.joblib")
            Z = _transform_until_model(b["pipeline"], te[b["features"]])
            expl = shap.TreeExplainer(b["pipeline"].named_steps["model"])
            sv = expl.shap_values(Z)
            shap.summary_plot(sv, Z, feature_names=b["selected_features"], show=False, max_display=15)
            fig = plt.gcf()
            fig.suptitle(f"Fig. 12d. SHAP summary, {_label(tree_best)} (2025 test)")
            _save(fig, fdir / "12d_shap_summary.png")
            out["shap"] = f"computed for {tree_best}"
        except ImportError:
            out["shap"] = "skipped (shap not installed; optional)"
        except Exception as exc:  # pragma: no cover
            out["shap"] = f"skipped ({type(exc).__name__}: {exc})"
    top = (perm[perm.model == best].nlargest(10, "mae_increase_mean")[["feature", "mae_increase_mean"]]
           .to_dict(orient="records"))
    out["best_model"] = best
    out["top_permutation_features_best_model"] = top
    (mdir / "E6_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info("E6 interpretability done (%s)", out["shap"])
    return out

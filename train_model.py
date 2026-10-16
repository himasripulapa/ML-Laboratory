import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config_data import MODELS, build_preprocessor

# Train + tune + evaluate all four models
def run_experiment(X, y, class_names, cfg):
    counts = y.value_counts()
    n_rows, n_classes = len(X), len(counts)
    n_test = int(np.ceil(n_rows * cfg["test_size"]))
    notes = []
    knn_tuning = None
    # Too little data to split / cross-validate? Then train and evaluate on the same rows.
    tiny = (n_rows < 10 or counts.min() < 2
            or n_test < n_classes or (n_rows - n_test) < n_classes)
    if tiny:
        X_tr, X_te, y_tr, y_te = X, X, y, y
        n_splits = None
        cfg = {**cfg, "tune": False}
        notes.append(f"Only {n_rows} rows (smallest class has {counts.min()}). That is too little data to "
                     "split into train/test or cross-validate, so every model was trained AND evaluated on "
                     "the same rows. Train and test scores are therefore identical and say how well each "
                     "model memorised the data, not how well it generalises. Hyperparameter tuning was "
                     "switched off. Use at least ~100 rows for meaningful results.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"])
        n_splits = int(max(2, min(5, y_tr.value_counts().min())))
    results = {}
    for name, (est, grid) in MODELS.items():
        model = clone(est)
        if tiny and name == "KNN":
            model.set_params(n_neighbors=max(1, min(5, len(X_tr) - 1)))
        pipe = Pipeline([("prep", build_preprocessor(X, cfg["impute"], cfg["scale"])),
                         ("model", model)])

        t0 = time.time()

        if cfg["tune"]:
            gs = GridSearchCV(pipe, grid, cv=n_splits, scoring="f1_weighted",
                              n_jobs=-1, error_score=np.nan)
            gs.fit(X_tr, y_tr)
            best, cv_f1 = gs.best_estimator_, gs.best_score_

            # Save KNN tuning results for display
            if name == "KNN":
                knn_tuning = {
                    "optimal_k": gs.best_params_["model__n_neighbors"],
                    "weights": gs.best_params_["model__weights"],
                    "cv_f1": gs.best_score_,
                }
        elif tiny:
            best, cv_f1 = pipe.fit(X_tr, y_tr), np.nan
        else:
            best = pipe.fit(X_tr, y_tr)
            cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=n_splits,
                                    scoring="f1_weighted", error_score=np.nan).mean()

        elapsed = time.time() - t0

        pred = best.predict(X_te)
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="weighted", zero_division=0)

        results[name] = dict(
            pipe=best, pred=pred, params={k[7:]: best.get_params()[k] for k in grid},
            metrics={"Train Acc": best.score(X_tr, y_tr), "Test Acc": best.score(X_te, y_te),
                     "Precision": p, "Recall": r, "F1": f1, "CV F1": cv_f1, "Time (s)": elapsed})
    counts = y.value_counts()
    meta = dict(n_rows=len(X), n_feat_raw=X.shape[1],
                n_feat_t=results["KNN"]["pipe"].named_steps["prep"].transform(X_te.iloc[:1]).shape[1],
                imbalance=counts.max() / counts.min(), n_classes=len(class_names),
                missing_pct=X.isna().mean().mean() * 100)
    return dict(results=results, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                cfg=cfg, meta=meta, class_names=class_names, n_splits=n_splits,
                tiny=tiny, notes=notes, knn_tuning=knn_tuning)

# Decision boundaries (project to 2-D, refit each model, paint the plane)

def plot_boundaries(exp, mode, feat_pair=None):
    cfg = exp["cfg"]
    pre = build_preprocessor(exp["X_tr"], cfg["impute"], cfg["scale"]).fit(exp["X_tr"])
    Z_tr, Z_te = pre.transform(exp["X_tr"]), pre.transform(exp["X_te"])
    if mode == "PCA (2 components)":
        pca = PCA(2).fit(Z_tr)
        Z_tr, Z_te = pca.transform(Z_tr), pca.transform(Z_te)
        xl, yl = "PC1", "PC2"
    else:
        names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
        i, j = names.index(feat_pair[0]), names.index(feat_pair[1])
        Z_tr, Z_te = Z_tr[:, [i, j]], Z_te[:, [i, j]]
        xl, yl = feat_pair

    n = len(exp["class_names"])
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colors = [cmap(i) for i in range(n)]
    allz = np.vstack([Z_tr, Z_te])
    xx, yy = np.meshgrid(np.linspace(allz[:, 0].min() - .5, allz[:, 0].max() + .5, 200),
                         np.linspace(allz[:, 1].min() - .5, allz[:, 1].max() + .5, 200))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, r) in zip(axes.ravel(), exp["results"].items()):
        est = clone(r["pipe"].named_steps["model"]).fit(Z_tr, exp["y_tr"])
        zz = est.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=np.arange(-.5, n + .5, 1), colors=colors, alpha=.25)
        ax.scatter(Z_tr[:, 0], Z_tr[:, 1], c=[colors[k] for k in exp["y_tr"]], s=18, alpha=.6)
        ax.scatter(Z_te[:, 0], Z_te[:, 1], c=[colors[k] for k in exp["y_te"]],
                   s=45, edgecolor="black", marker="s", linewidth=.8)
        ax.set_title(f"{name}  (2-D test acc {est.score(Z_te, exp['y_te']):.2f})")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.suptitle("Circles = training points, squares = test points", y=1.0, fontsize=10)
    fig.tight_layout()
    return fig
# Rule-based explanations of where each algorithm wins / fails

def build_insights(exp):
    res, m, cfg = exp["results"], exp["meta"], exp["cfg"]
    f1 = {k: v["metrics"]["F1"] for k, v in res.items()}
    best = max(f1, key=f1.get)
    gap = {k: v["metrics"]["Train Acc"] - v["metrics"]["Test Acc"] for k, v in res.items()}
    out = {k: [] for k in res}
    out["KNN"] += (["⚠️ Scaling was OFF; KNN relies on distances, so big-range features dominate."]
                   if not cfg["scale"] else [])
    if m["n_feat_t"] > 30:
        out["KNN"].append(f"⚠️ {m['n_feat_t']} features after preprocessing: curse of dimensionality weakens distances.")
    if gap["KNN"] > .1:
        out["KNN"].append(f"Train/test gap {gap['KNN']:.2f} - memorising noise (k={res['KNN']['params']['n_neighbors']} may be too small).")
    if m["n_rows"] > 5000:
        out["KNN"].append("Prediction time grows with dataset size - KNN compares to every stored point.")
    if gap["Decision Tree"] > .1:
        out["Decision Tree"].append(f"⚠️ Overfitting: train acc beats test acc by {gap['Decision Tree']:.2f}. Try a smaller max_depth.")
    else:
        out["Decision Tree"].append("Generalises well here - depth/leaf limits kept it from memorising.")
    kern = res["SVM"]["params"]["kernel"]
    if kern == "rbf" and f1["SVM"] - f1["Logistic Regression"] > .03:
        out["SVM"].append("RBF beat the linear model: the classes need a curved boundary.")
    elif kern == "linear":
        out["SVM"].append("Tuning picked a linear kernel: classes are close to linearly separable.")
    if m["n_rows"] > 3000:
        out["SVM"].append(f"Training took {res['SVM']['metrics']['Time (s)']:.1f}s - SVM scales poorly beyond tens of thousands of rows.")
    diff = f1[best] - f1["Logistic Regression"]
    if diff < .02:
        out["Logistic Regression"].append("Matches the best model: a straight line is enough, so simple wins.")
    elif diff > .05:
        out["Logistic Regression"].append(f"⚠️ {diff:.2f} F1 behind the best model: the data has non-linear structure a line cannot capture.")
    if not cfg["scale"]:
        out["Logistic Regression"].append("Scaling OFF can slow convergence and distort regularisation.")
    out[best].insert(0, f"🏆 Best F1 on the test set ({f1[best]:.3f}).")
    glob = []
    if m["imbalance"] > 3:
        glob.append(f"Class imbalance is {m['imbalance']:.1f}:1 - accuracy can look great while minority classes fail. Trust F1 and the confusion matrices.")
    if m["missing_pct"] > 0:
        glob.append(f"{m['missing_pct']:.1f}% of feature cells were missing and were imputed ({cfg['impute']}) inside the pipeline, fit on training data only.")
    return out, glob
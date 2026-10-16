import warnings
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
from config_data import (HOW_THEY_THINK, MODELS, SWEEPS, build_preprocessor,
                         inject_missing, load_sample, split_columns)
from train_model import build_insights, plot_boundaries, run_experiment
warnings.filterwarnings("ignore")
st.set_page_config(page_title="ML Laboratory", page_icon="🧪", layout="wide",
                   initial_sidebar_state="expanded")

# ---- Look & feel -----------------------------------------------------------
PRIMARY = "#4f46e5"
ACCENT = "#06b6d4"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.titleweight": "bold", "axes.titlesize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "grid.color": "#e2e8f0",
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.5rem; max-width: 1280px; }}

.hero {{
  background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
  border-radius: 18px; padding: 28px 32px; color: white; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(79,70,229,.25);
}}
.hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 700; color: white; }}
.hero p {{ margin: 6px 0 0; opacity: .92; font-size: 1.02rem; }}
.chips {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{
  background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  border-radius: 999px; padding: 4px 12px; font-size: .82rem; font-weight: 500;
}}

[data-testid="stMetric"] {{
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
  padding: 14px 18px; box-shadow: 0 1px 3px rgba(15,23,42,.05);
}}
[data-testid="stMetricLabel"] {{ color: #64748b; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: #0f172a; font-weight: 700; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #e2e8f0; }}
.stTabs [data-baseweb="tab"] {{
  border-radius: 10px 10px 0 0; padding: 10px 16px; font-weight: 600; color: #475569;
}}
.stTabs [aria-selected="true"] {{ color: {PRIMARY}; background: #eef2ff; }}

section[data-testid="stSidebar"] {{ background: #f8fafc; border-right: 1px solid #e2e8f0; }}
section[data-testid="stSidebar"] h2 {{
  font-size: 1rem; text-transform: uppercase; letter-spacing: .06em; color: {PRIMARY};
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {PRIMARY}, {ACCENT}); border: 0; font-weight: 600;
  border-radius: 10px; padding: .6rem 1rem;
}}
.stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); }}

.empty {{
  border: 2px dashed #c7d2fe; border-radius: 16px; padding: 36px; text-align: center;
  background: #f5f7ff; color: #3730a3;
}}
.empty h3 {{ margin: 0 0 6px; }}
.winner {{
  background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 14px;
  padding: 14px 20px; color: #065f46; margin-bottom: 14px; font-size: 1.02rem;
}}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 ML Laboratory")
    st.caption("Configure → Run → Explore")

    st.header("1 · Data")
    source = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer", "Moons (synthetic)",
                                      "Circles (synthetic)", "Upload my CSV"])
    if source == "Upload my CSV":
        up = st.file_uploader("CSV file", type="csv")
        if up is None:
            st.info("Upload a CSV to begin (needs one categorical target column).")
            st.stop()
        df = pd.read_csv(up)
    else:
        df = load_sample(source)
    target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
    tgt_continuous = (pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20)
    bin_target = False
    if tgt_continuous:
        bin_target = st.checkbox("Target is continuous: convert to Low / Medium / High",
                                 value=True,
                                 help="Splits the numbers into 3 equal-sized groups so the lab can classify them.")
    if source != "Upload my CSV":
        miss = st.slider("Inject random missing values (%)", 0, 30, 0,
                         help="Demo only: punches holes in the data so you can see imputation work.")
        if miss:
            df = inject_missing(df, target, miss)
    id_like = [c for c in df.columns if c != target and df[c].dtype == object and df[c].nunique() == len(df)]
    drop_cols = st.multiselect("Columns to ignore (IDs, names)", [c for c in df.columns if c != target],
                               default=id_like)

    st.header("2 · Preprocessing")
    impute = st.selectbox("Numeric imputation", ["median", "mean", "most_frequent"])
    scale = st.checkbox("Standardise numeric features", True)

    st.header("3 · Experiment")
    test_size = st.slider("Test set size", 0.1, 0.5, 0.25, 0.05)
    tune = st.checkbox("Tune hyperparameters (GridSearchCV)", True)
    seed = st.number_input("Random seed", 0, 9999, 42)
    st.divider()
    run = st.button("▶ Run experiment", type="primary", width="stretch")

# ---- prepare X / y ---------------------------------------------------------
data = df.dropna(subset=[target]).copy()
y_raw = data[target]
problems = []
if tgt_continuous and bin_target and len(data) >= 3:
    y_raw = pd.qcut(y_raw.rank(method="first"), 3, labels=["Low", "Medium", "High"]).astype(str)
elif tgt_continuous:
    problems.append(f"'{target}' looks continuous ({y_raw.nunique()} unique values). "
                    "Tick the box in the sidebar to convert it to Low / Medium / High, or pick another target.")
X = data.drop(columns=[target] + drop_cols).dropna(axis=1, how="all")
if len(X) == 0:
    problems.append("No rows have a value in the target column.")
if X.shape[1] == 0:
    problems.append("No feature columns are left. Un-ignore some columns in the sidebar.")
if y_raw.nunique() < 2:
    problems.append(f"The target '{target}' has fewer than 2 different values, so there is nothing to classify. "
                    "Choose a different target column or add more rows.")
can_train = not problems

# ---- Hero ------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🧪 ML Laboratory</h1>
  <p>See how KNN, Decision Tree, SVM and Logistic Regression each <i>think</i> about your data.</p>
  <div class="chips">
    <span class="chip">📁 {source}</span>
    <span class="chip">🎯 {target}</span>
    <span class="chip">🧹 {impute} · {'scaled' if scale else 'unscaled'}</span>
    <span class="chip">🧪 {int(test_size * 100)}% test</span>
  </div>
</div>
""", unsafe_allow_html=True)

for p_ in problems:
    st.warning(p_ + " The data and preprocessing tabs still work; training is skipped.")
if len(X) > 5000:
    idx = X.sample(5000, random_state=0).index
    X, y_raw = X.loc[idx], y_raw.loc[idx]
    st.warning("Dataset capped at 5,000 random rows to keep the lab responsive.")
le = LabelEncoder().fit(y_raw.astype(str))
y = pd.Series(le.transform(y_raw.astype(str)), index=X.index)
class_names = list(le.classes_)

if run and not can_train:
    st.session_state.pop("exp", None)
if run and can_train:
    with st.spinner("Preprocessing, tuning and training four models..."):
        st.session_state["exp"] = run_experiment(
            X, y, class_names,
            dict(test_size=test_size, impute=impute, scale=scale, tune=tune, seed=int(seed)))
        st.session_state["dataset_name"] = source
    st.toast("Experiment finished", icon="✅")
exp = st.session_state.get("exp")

tabs = st.tabs(["📋 Data", "🧹 Preprocessing", "📊 Metrics",
                "🗺️ Boundaries", "💡 Explanation", "🔬 Experiments"])


def need_results():
    if can_train:
        st.markdown("""<div class="empty"><h3>No results yet</h3>
        Press <b>▶ Run experiment</b> in the sidebar to train the four models.</div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div class="empty"><h3>Training not possible</h3>
        See the warning at the top of the page for what to fix.</div>""",
                    unsafe_allow_html=True)


def show_notes(exp):
    for n_ in exp.get("notes", []):
        st.warning(n_)


# ---- Tab 1 ------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(X):,}")
    c2.metric("Features", X.shape[1])
    c3.metric("Classes", len(class_names))
    c4.metric("Missing cells", f"{int(X.isna().sum().sum()):,}")

    st.subheader("Preview")
    st.dataframe(df.head(10), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values")
        mv = pd.DataFrame({"dtype": X.dtypes.astype(str), "missing": X.isna().sum(),
                           "missing %": (X.isna().mean() * 100).round(2)})
        if mv["missing"].sum() == 0:
            st.success("No missing values in the features.")
        else:
            st.dataframe(mv[mv["missing"] > 0], width="stretch")
            st.bar_chart(mv["missing %"], color=PRIMARY)
        n_miss_target = int(df[target].isna().sum())
        if n_miss_target:
            st.warning(f"{n_miss_target} rows have no target label and were dropped.")
    with right:
        if can_train:
            st.subheader("Class distribution")
            st.bar_chart(y_raw.astype(str).value_counts(), color=ACCENT)

# ---- Tab 2 ------------------------------------------------------------------
with tabs[1]:
    num_cols, cat_cols = split_columns(X)
    st.subheader("Pipeline")
    st.caption("Fit on the training split only, so there is no data leakage. Applied to every model.")
    s1, s2, s3 = st.columns(3)
    with s1.container(border=True):
        st.markdown(f"**1 · Numeric** ({len(num_cols)})")
        st.write(f"Impute with **{impute}**" + (" → standardise" if scale else ""))
    with s2.container(border=True):
        st.markdown(f"**2 · Categorical** ({len(cat_cols)})")
        st.write("Impute **most frequent** → one-hot encode")
    with s3.container(border=True):
        st.markdown("**3 · Model**")
        st.write("KNN · Decision Tree · SVM · Logistic Regression")
    with st.expander("Column types", expanded=False):
        st.write("**Numeric:**", ", ".join(num_cols) or "-")
        st.write("**Categorical:**", ", ".join(cat_cols) or "-")
    if exp:
        pre = exp["results"]["KNN"]["pipe"].named_steps["prep"]
        st.success(f"{exp['meta']['n_feat_raw']} raw features → {exp['meta']['n_feat_t']} model-ready features.")
        Zs = pd.DataFrame(pre.transform(exp["X_te"].head(8)),
                          columns=[n.split("__", 1)[1] for n in pre.get_feature_names_out()])
        st.caption("First rows of the transformed test data:")
        st.dataframe(Zs, width="stretch")
    else:
        st.info("Run the experiment to see the transformed data.")

# ---- Tab 3 ------------------------------------------------------------------
with tabs[2]:
    if not exp:
        need_results()
    else:
        show_notes(exp)
        mt = pd.DataFrame({k: v["metrics"] for k, v in exp["results"].items()}).T
        best = mt["F1"].idxmax()
        st.markdown(f'<div class="winner">🏆 Best model by F1: <b>{best}</b> '
                    f'(F1 {mt.loc[best, "F1"]:.3f}, test accuracy {mt.loc[best, "Test Acc"]:.3f})</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(mt))
        for col, (name, row) in zip(cols, mt.iterrows()):
            col.metric(name, f"{row['Test Acc']:.3f}", help="Test accuracy")
        st.dataframe(mt.style.format("{:.3f}").highlight_max(
            subset=["Test Acc", "Precision", "Recall", "F1", "CV F1"], color="#bbf7d0"),
            width="stretch")
        st.download_button("⬇ Download metrics (CSV)", mt.to_csv().encode(),
                           file_name="ml_lab_metrics.csv", mime="text/csv")
        c1, c2 = st.columns(2)
        c1.subheader("Test metrics")
        c1.bar_chart(mt[["Test Acc", "Precision", "Recall", "F1"]])
        c2.subheader("Train vs test accuracy")
        c2.caption("A big gap means overfitting.")
        c2.bar_chart(mt[["Train Acc", "Test Acc"]])
        with st.expander("Chosen hyperparameters"):
            st.dataframe(pd.DataFrame({k: {p: str(v) for p, v in r["params"].items()}
                                       for k, r in exp["results"].items()}).T.fillna("-"),
                         width="stretch")
        st.subheader("Confusion matrices")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (name, r) in zip(axes.ravel(), exp["results"].items()):
            ConfusionMatrixDisplay.from_predictions(exp["y_te"], r["pred"],
                                                    labels=list(range(len(exp["class_names"]))),
                                                    display_labels=exp["class_names"],
                                                    ax=ax, colorbar=False, cmap="Purples")
            ax.set_title(name)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---- Tab 4 ------------------------------------------------------------------
with tabs[3]:
    if not exp:
        need_results()
    else:
        st.write("Each model is refit on a **2-D view** of the data so its decision regions can be drawn. "
                 "Shaded areas show what the model predicts at every point of the plane.")
        mode = st.radio("2-D view", ["PCA (2 components)", "Pick two features"], horizontal=True)
        pair = None
        if mode == "Pick two features":
            pre = build_preprocessor(exp["X_tr"], exp["cfg"]["impute"], exp["cfg"]["scale"]).fit(exp["X_tr"])
            names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
            if len(names) < 2:
                st.error("Need at least two features.")
                mode = "PCA (2 components)"
            else:
                a, b = st.columns(2)
                f1_ = a.selectbox("X axis", names, index=0)
                f2_ = b.selectbox("Y axis", names, index=1)
                pair = (f1_, f2_)
        show_notes(exp)
        try:
            fig = plot_boundaries(exp, mode, pair)
        except Exception as err:
            fig = None
            st.warning("Not enough data or features to draw a 2-D picture "
                       f"(PCA needs at least 2 features and 2 rows). Details: {err}")
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        k1, k2, k3, k4 = st.columns(4)
        k1.info("**KNN**\n\nJagged and local")
        k2.info("**Decision Tree**\n\nRectangles")
        k3.info("**Logistic Regression**\n\nStraight lines")
        k4.info("**SVM (RBF)**\n\nSmooth curves")

# ---- Tab 5 ------------------------------------------------------------------
with tabs[4]:
    if not exp:
        need_results()
    else:
        show_notes(exp)
        insights, glob = build_insights(exp)
        for g in glob:
            st.info(g)
        cols = st.columns(2)
        for i, (name, (tagline, how, good, bad)) in enumerate(HOW_THEY_THINK.items()):
            with cols[i % 2].container(border=True):
                m = exp["results"][name]["metrics"]
                st.subheader(f"{name}")
                st.caption(tagline)
                a, b, c = st.columns(3)
                a.metric("Test acc", f"{m['Test Acc']:.3f}")
                b.metric("F1", f"{m['F1']:.3f}")
                c.metric("Train–test gap", f"{m['Train Acc'] - m['Test Acc']:+.3f}")
                st.write(how)
                st.markdown(f"**✅ Succeeds on:** {good}\n\n**❌ Struggles with:** {bad}")
                if insights[name]:
                    st.markdown("**On your data:**")
                    for line in insights[name]:
                        st.markdown(f"- {line}")
        with st.expander("🌳 Peek inside the Decision Tree (top 3 levels)"):
            tree_pipe = exp["results"]["Decision Tree"]["pipe"]
            fnames = [n.split("__", 1)[1] for n in tree_pipe.named_steps["prep"].get_feature_names_out()]
            st.code(export_text(tree_pipe.named_steps["model"], feature_names=fnames, max_depth=3))

# ---- Tab 6 ------------------------------------------------------------------
with tabs[5]:
    if not exp:
        need_results()
    elif exp["tiny"]:
        st.info("The hyperparameter sweep needs enough rows for cross-validation (about 10 or more, "
                "with at least 2 rows per class). Upload a larger file to use this tab.")
    else:
        st.write("Sweep one hyperparameter and watch the **bias-variance trade-off**: "
                 "train score keeps climbing while validation score peaks, then drops.")
        name = st.selectbox("Model", list(MODELS))
        param, values, logx = SWEEPS[name]
        if name == "KNN":
            cap = int(len(exp["X_tr"]) * (exp["n_splits"] - 1) / exp["n_splits"])
            values = [v for v in values if v < cap]
        pipe = clone(exp["results"][name]["pipe"])
        with st.spinner("Running sweep..."):
            tr, va = validation_curve(pipe, exp["X_tr"], exp["y_tr"], param_name=param, param_range=values,
                                      cv=exp["n_splits"], scoring="f1_weighted", n_jobs=-1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(values, tr.mean(1), "o-", color=PRIMARY, label="Train F1")
        ax.plot(values, va.mean(1), "s-", color=ACCENT, label="Validation F1")
        ax.fill_between(values, va.mean(1) - va.std(1), va.mean(1) + va.std(1),
                        color=ACCENT, alpha=.15)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(param.replace("model__", ""))
        ax.set_ylabel("Weighted F1")
        ax.legend(frameon=False)
        ax.grid(alpha=.5)
        st.pyplot(fig)
        plt.close(fig)
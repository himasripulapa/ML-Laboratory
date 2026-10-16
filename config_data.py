import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_circles, make_moons
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# The four algorithms + the hyperparameter grids we will search

MODELS = {
    "KNN": (KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7, 11, 15],
             "model__weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42),
                      {"model__max_depth": [2, 3, 5, 8, None],
                       "model__min_samples_leaf": [1, 3, 5, 10]}),
    "SVM": (SVC(random_state=42),
            {"model__C": [0.1, 1, 10],
             "model__kernel": ["linear", "rbf"]}),
    "Logistic Regression": (LogisticRegression(max_iter=2000),
                            {"model__C": [0.01, 0.1, 1, 10]}),
}

# parameter to sweep in the "Experiments" tab
SWEEPS = {
    "KNN": ("model__n_neighbors", [1, 3, 5, 7, 9, 11, 15, 21, 31], False),
    "Decision Tree": ("model__max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 15, 20], False),
    "SVM": ("model__C", [0.01, 0.1, 1, 10, 100], True),
    "Logistic Regression": ("model__C", [0.001, 0.01, 0.1, 1, 10, 100], True),
}

HOW_THEY_THINK = {
    "KNN": ("🏘️ Asks the neighbours",
            "Stores every training point. To classify a new point it finds the k closest ones and takes a vote.",
            "Irregular, local patterns; small, low-dimensional data.",
            "Many features (distances lose meaning), unscaled features, huge data (slow predictions), noisy labels with tiny k."),
    "Decision Tree": ("🌳 Plays 20 questions",
                      "Repeatedly asks yes/no questions on one feature ('petal length < 2.4?') to split the data into pure groups.",
                      "Mixed feature types, interactions between features, needs no scaling, easy to explain.",
                      "Overfits if grown deep; boundaries are axis-aligned boxes, so diagonal or curved boundaries need many splits."),
    "SVM": ("🛣️ Builds the widest road",
            "Finds the boundary with the largest margin between classes. The RBF kernel bends this boundary into curves.",
            "Medium-sized data, clear margins, non-linear shapes (RBF), high-dimensional data.",
            "Very large datasets (slow), unscaled features, heavily overlapping classes, needs careful C / gamma."),
    "Logistic Regression": ("📏 Draws one straight line",
                            "Learns a weighted sum of features and squashes it into a probability. The boundary is a straight line/plane.",
                            "Roughly linearly separable data, many features, needs a fast and interpretable baseline.",
                            "Curved or XOR-like patterns - it cannot bend its boundary."),
}



# Data loading helpers

@st.cache_data
def load_sample(name):
    if name in ("Iris", "Wine", "Breast Cancer"):
        d = {"Iris": datasets.load_iris, "Wine": datasets.load_wine,
             "Breast Cancer": datasets.load_breast_cancer}[name](as_frame=True)
        df = d.frame.copy()
        df["target"] = pd.Series(d.target).map(lambda i: str(d.target_names[i])).values
        return df
    if name == "Moons (synthetic)":
        X, y = make_moons(400, noise=0.3, random_state=0)
    else:
        X, y = make_circles(400, noise=0.15, factor=0.5, random_state=0)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["target"] = y
    return df


def inject_missing(df, target, pct, seed=0):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in out.columns.drop(target):
        out.loc[rng.random(len(out)) < pct / 100, c] = np.nan
    return out

# Preprocessing (impute -> scale -> one-hot) inside a Pipeline
def split_columns(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def build_preprocessor(X, num_strategy, scale):
    num_cols, cat_cols = split_columns(X)
    num_steps = [("imputer", SimpleImputer(strategy=num_strategy))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols),
                              ("cat", Pipeline(cat_steps), cat_cols)])
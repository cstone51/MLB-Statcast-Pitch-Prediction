import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    log_loss, accuracy_score, f1_score, top_k_accuracy_score
)
from sklearn.preprocessing import LabelEncoder

from .data import pull_data, clean_data
from .features import feature_engineering, build_feature_pipeline, build_features

MODEL_OUTPUT_PATH = Path("models/model.joblib")


# def build_pipeline(num_classes: int) -> Pipeline:
#     return Pipeline([
#         ("preprocessor", build_feature_pipeline()),
#         ("classifier", XGBClassifier(
#             objective="multi:softprob",   # optimizes for calibrated probabilities
#             num_class=num_classes,
#             n_estimators=400,
#             learning_rate=0.05,
#             max_depth=5,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             reg_lambda=1.0,
#             random_state=42,
#             n_jobs=-1,
#             eval_metric="mlogloss",
#         )),
#     ])

# def build_pipeline(num_classes: int) -> Pipeline:
#     return Pipeline([
#         ("preprocessor", build_feature_pipeline()),
#         ("classifier", RandomForestClassifier(
#             n_estimators=400,
#             max_depth=10,
#             min_samples_leaf=5,
#             class_weight="balanced",
#             random_state=42,
#             n_jobs=-1,
#         )),
#     ])

# def build_pipeline(num_classes: int) -> Pipeline:
#     return Pipeline([
#         ("preprocessor", build_feature_pipeline()),
#         ("classifier", KNeighborsClassifier(
#             n_neighbors=15,
#             weights="distance",   # closer neighbors vote more heavily
#             metric="euclidean",
#             n_jobs=-1,
#         )),
#     ])

from sklearn.linear_model import LogisticRegression

def build_pipeline(num_classes: int, class_weights: dict = None) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_feature_pipeline()),
        ("classifier", LogisticRegression(
            #multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            C=1.0,               # inverse regularization strength
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        )),
    ])


from sklearn.utils.class_weight import compute_class_weight

def custom_weights(y_train_enc: np.ndarray) -> dict:

    classes = np.unique(y_train_enc)
    counts  = np.bincount(y_train_enc)
    freqs   = counts / len(y_train_enc)

    raw_weights = 1.0 / freqs               # inverse frequency
    sqrt_weights = np.sqrt(raw_weights)     # soften with sqrt
    sqrt_weights /= sqrt_weights.min()      # normalize so most common class = 1.0

    return {cls: sqrt_weights[cls] for cls in classes}


def time_based_split(X, y, train_size=0.65):
    """
    Chronological split — preserves temporal order.
    Prevents future pitch data from leaking into the training set.
    """
    split_idx = int(len(X) * train_size)
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:]
    )


def filter_unseen_test_classes(X_test, y_test_raw, le: LabelEncoder):
    """
    Remove test rows whose class was never seen during training.
    Necessary after a time-based split — pitchers occasionally add/drop
    pitch types mid-season, so the test window may contain classes the
    encoder was never fit on.
    """
    seen = set(le.classes_)
    mask = y_test_raw.isin(seen)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} test rows with unseen classes: "
              f"{y_test_raw[~mask].unique().tolist()}")
    return X_test.loc[mask], y_test_raw.loc[mask]


def train(output_path: Path = MODEL_OUTPUT_PATH) -> Pipeline:
    """
    Full training run with chronological split and calibrated XGBoost.
    Logs all metrics to MLflow and saves model + label encoder to disk.
    """
    df = pull_data("Kevin", "Gausman", "2024-01-01", "2025-12-31")
    print(f"puling data... {len(df)} rows")
    df = clean_data(df)
    df = feature_engineering(df)
    X, y = build_features(df)

    # --- chronological split (no shuffling) ---
    X_train, X_test, y_train_raw, y_test_raw = time_based_split(X, y, train_size=0.65)
    print()
    # --- encode labels on train set only ---
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)

    # --- filter test rows with classes unseen in training ---
    X_test, y_test_raw = filter_unseen_test_classes(X_test, y_test_raw, le)
    y_test_enc = le.transform(y_test_raw)

    # --- labels array — single source of truth from the encoder ---
    n_classes          = len(le.classes_)
    labels_for_metrics = np.arange(n_classes)

    print(f"Classes: {le.classes_.tolist()}")
    print(f"Train size: {len(y_train_enc)} | Test size: {len(y_test_enc)}")

    class_weights = custom_weights(y_train_enc)
    print("Class weights:")
    for cls, w in class_weights.items():
        print(f"  {le.classes_[cls]}: {w:.3f}")

# ...


    with mlflow.start_run():
        pipeline = build_pipeline(num_classes=n_classes, class_weights=class_weights)
        pipeline.fit(X_train, y_train_enc)

        y_pred_enc = pipeline.predict(X_test)
        y_proba    = pipeline.predict_proba(X_test)          # full matrix (n_samples, n_classes)
        y_pred     = le.inverse_transform(y_pred_enc)        # back to pitch type strings

        # --- metrics ---
        acc   = accuracy_score(y_test_enc, y_pred_enc)
        f1_w  = f1_score(y_test_enc, y_pred_enc, average="weighted")
        f1_m  = f1_score(y_test_enc, y_pred_enc, average="macro")
        ll    = log_loss(y_test_enc, y_proba, labels=labels_for_metrics)
        auc   = roc_auc_score(y_test_enc, y_proba, multi_class="ovr",
                              average="weighted", labels=labels_for_metrics)
        top2  = top_k_accuracy_score(y_test_enc, y_proba, k=2,
                                     labels=labels_for_metrics)

        # --- baseline: always predict training class proportions ---
        base_probs     = np.bincount(y_train_enc, minlength=n_classes) / len(y_train_enc)
        baseline_probs = np.tile(base_probs, (len(y_test_enc), 1))
        baseline_ll    = log_loss(y_test_enc, baseline_probs, labels=labels_for_metrics)
        improvement    = baseline_ll - ll

        # --- log to mlflow ---
        mlflow.log_params({
            "n_estimators":    400,
            "learning_rate":   0.05,
            "max_depth":       5,
            "subsample":       0.9,
            "colsample_bytree": 0.9,
            "reg_lambda":      1.0,
            "train_size":      0.65,
            "n_classes":       n_classes,
        })
        mlflow.log_metrics({
            "log_loss":          ll,
            "baseline_log_loss": baseline_ll,
            "ll_improvement":    improvement,
            "roc_auc":           auc,
            "accuracy":          acc,
            "f1_weighted":       f1_w,
            "f1_macro":          f1_m,
            "top2_accuracy":     top2,
        })
        mlflow.sklearn.log_model(pipeline, "model")

        # --- print summary ---
        present_classes = sorted(y_test_raw.unique())
        print(classification_report(y_test_raw, y_pred, target_names=present_classes))
        print(f"Accuracy:    {acc:.4f}")
        print(f"Top-2 Acc:   {top2:.4f}")
        print(f"ROC-AUC:     {auc:.4f}")
        print(f"Log Loss:    {ll:.4f}  (Baseline: {baseline_ll:.4f}, "
              f"Improvement: {improvement:.4f})")

    # --- save artifacts ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    joblib.dump(le, output_path.parent / "label_encoder.joblib")
    print(f"Model saved to {output_path}")
    return pipeline
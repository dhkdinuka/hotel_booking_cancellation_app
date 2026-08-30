from pathlib import Path
import json

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/hotel_bookings.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_canceled"

FEATURES = [
    "hotel",
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

NUMERIC_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]


def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(name, pipeline, X_test, y_test):
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(
            precision_score(y_test, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, proba)
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            pred,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    print(f"Using pandas version: {pd.__version__}")
    print(f"Using scikit-learn version: {sklearn.__version__}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download hotel_bookings.csv and place it "
            "in the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        column
        for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    df = df[FEATURES + [TARGET]].copy()

    # Remove rows where no guests are present.
    total_guests = (
        df["adults"].fillna(0)
        + df["children"].fillna(0)
        + df["babies"].fillna(0)
    )

    df = df[total_guests > 0].copy()

    # Keep sensible ADR values while retaining missing values
    # for the imputer to handle.
    df = df[
        df["adr"].isna()
        | (
            (df["adr"] >= 0)
            & (df["adr"] <= 1000)
        )
    ].copy()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=3,
            random_state=42,
        ),
    }

    results = {}
    fitted = {}

    for name, model in candidates.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results[name] = metrics
        fitted[name] = pipeline

        print(
            f"{name}: "
            f"F1={metrics['f1']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

    best_name = max(
        results,
        key=lambda name: results[name]["f1"],
    )

    best_model = fitted[best_name]

    model_path = (
        MODEL_DIR
        / "hotel_cancellation_model.joblib"
    )

    metadata_path = (
        MODEL_DIR
        / "metadata.json"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    metadata = {
        "best_model": best_name,
        "selection_metric": "f1",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "results": results,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
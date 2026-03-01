import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TARGET_COL = "AvgRating"
FEATURE_COLUMNS = [
    "mech_roll_spin_and_move",
    "GameWeight",
    "Cat_War",
    "YearPublished",
]


def find_training_csv(train_dir: str) -> str:
    csv_files = [f for f in os.listdir(train_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {train_dir}")
    csv_files.sort()
    return os.path.join(train_dir, csv_files[0])


if __name__ == "__main__":
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = find_training_csv(train_dir)
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required = FEATURE_COLUMNS + [TARGET_COL]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from training data: {missing_required}")

    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    for c in FEATURE_COLUMNS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    spearman = float(pd.Series(y_test).corr(pd.Series(preds), method="spearman"))

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")

    feature_path = os.path.join(model_dir, "feature_columns.json")
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": FEATURE_COLUMNS}, f, indent=2)
    print(f"Saved feature list to: {feature_path}")

    metrics = {
        "rows": int(len(X)),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": spearman,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")
    print(metrics)
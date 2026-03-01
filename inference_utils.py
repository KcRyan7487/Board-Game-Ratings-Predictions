import json
from pathlib import Path

import joblib
import pandas as pd


def load_feature_columns(feature_columns_path: Path) -> list[str]:
    raw = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        cols = raw.get("feature_columns", [])
    elif isinstance(raw, list):
        cols = raw
    else:
        cols = []
    if not cols:
        raise ValueError(f"No feature columns found in {feature_columns_path}")
    return cols


def load_artifacts(
    model_path: Path = Path("artifacts/model.pkl"),
    feature_columns_path: Path = Path("artifacts/feature_columns.json"),
):
    model = joblib.load(model_path)
    feature_columns = load_feature_columns(feature_columns_path)
    return model, feature_columns


def build_ordered_input_df(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    # Fill missing inputs with 0 and force numeric for robust inference from HTML form values.
    row = {col: payload.get(col, 0) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


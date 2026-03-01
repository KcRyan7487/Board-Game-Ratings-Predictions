import json
from pathlib import Path

import joblib
import pandas as pd


def load_feature_columns(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        cols = raw.get("feature_columns", [])
    elif isinstance(raw, list):
        cols = raw
    else:
        cols = []
    if not cols:
        raise ValueError(f"No feature columns found in {path}")
    return cols


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    model_path = repo / "artifacts" / "model.pkl"
    features_path = repo / "artifacts" / "feature_columns.json"
    data_path = repo / "data" / "processed" / "bgg_master.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature columns file: {features_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file for validation: {data_path}")

    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        raise TypeError("model.pkl is not a full sklearn Pipeline with a preprocessor step.")

    feature_columns = load_feature_columns(features_path)

    # Build one raw-input row and intentionally scramble columns first.
    df = pd.read_csv(data_path)
    # Recreate engineered columns if they are expected by the model.
    engineered = {}
    if "num_mechanics_active" in feature_columns and "num_mechanics_active" not in df.columns:
        mech_cols = [c for c in df.columns if c.startswith("mech_")]
        if mech_cols:
            engineered["num_mechanics_active"] = df[mech_cols].fillna(0).sum(axis=1).astype(int)
    if "num_subcats_active" in feature_columns and "num_subcats_active" not in df.columns:
        subcat_cols = [c for c in df.columns if c.startswith("subcat_")]
        if subcat_cols:
            engineered["num_subcats_active"] = df[subcat_cols].fillna(0).sum(axis=1).astype(int)
    if engineered:
        df = pd.concat([df, pd.DataFrame(engineered, index=df.index)], axis=1)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from source dataframe: {missing[:10]}")

    sample = df.loc[[0], feature_columns].copy()
    scrambled = sample[feature_columns[::-1]].copy()
    aligned = scrambled.reindex(columns=feature_columns).copy()

    # Raw values are passed directly; scaler/preprocessing happens inside pipeline.predict().
    pred = pipeline.predict(aligned)

    # Explicit hardcoded example: Terra Mystica (BGGId 120677)
    terra_payload = {col: 0 for col in feature_columns}
    terra_inputs = {
        "YearPublished": 2012,
        "GameWeight": 3.9666,
        "Cat_War": 0,
        "mech_roll_spin_and_move": 0,
    }
    for k, v in terra_inputs.items():
        if k in terra_payload:
            terra_payload[k] = v

    terra_df = pd.DataFrame([terra_payload], columns=feature_columns)
    terra_pred = pipeline.predict(terra_df)

    print("Inference contract validation passed.")
    print(f"Pipeline type: {type(pipeline).__name__}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Prediction sample: {float(pred[0]):.6f}")
    print(f"Prediction (Terra Mystica hardcoded input): {float(terra_pred[0]):.6f}")

    # Additional context: Show the actual AvgRating from the dataset for Terra Mystica BGGId 120677 too
    if "BGGId" in df.columns and "AvgRating" in df.columns:
        terra_rows = df.loc[df["BGGId"] == 120677, "AvgRating"]
        if not terra_rows.empty:
            print(f"Actual AvgRating for BGGId 120677 in dataset: {float(terra_rows.iloc[0]):.6f}")


if __name__ == "__main__":
    main()

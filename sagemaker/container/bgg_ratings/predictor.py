import io
import json
import os

import joblib
import pandas as pd
from flask import Flask, Response, request

DEFAULT_FEATURE_COLUMNS = [
    "mech_roll_spin_and_move",
    "GameWeight",
    "Cat_War",
    "YearPublished",
]

MODEL_PATH = os.path.join("/opt/ml/model", "model.pkl")
FEATURE_COLUMNS_PATH = os.path.join("/opt/ml/model", "feature_columns.json")

app = Flask(__name__)
model = None
feature_columns = DEFAULT_FEATURE_COLUMNS


def load_artifacts():
    global model, feature_columns
    if model is None:
        model = joblib.load(MODEL_PATH)
        if os.path.exists(FEATURE_COLUMNS_PATH):
            with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                feature_columns = raw.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
            elif isinstance(raw, list):
                feature_columns = raw
    return model, feature_columns


def parse_payload(raw_payload: str, content_type: str, cols: list[str]) -> pd.DataFrame:
    if content_type and "application/json" in content_type:
        body = json.loads(raw_payload)
        if isinstance(body, dict):
            df = pd.DataFrame([body])
        elif isinstance(body, list):
            if len(body) > 0 and isinstance(body[0], dict):
                df = pd.DataFrame(body)
            else:
                df = pd.DataFrame(body, columns=cols)
        else:
            raise ValueError("Unsupported JSON payload")
    else:
        df = pd.read_csv(io.StringIO(raw_payload), header=None)
        if df.shape[1] == len(cols):
            df.columns = cols

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return df[cols]


@app.route("/ping", methods=["GET"])
def ping():
    try:
        load_artifacts()
        return Response(response="\n", status=200, mimetype="application/json")
    except Exception:
        return Response(response="\n", status=404, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    mdl, cols = load_artifacts()
    payload = request.data.decode("utf-8")
    content_type = request.content_type or "text/csv"
    frame = parse_payload(payload, content_type, cols)
    preds = mdl.predict(frame)
    return Response(
        response=json.dumps(preds.tolist()),
        status=200,
        mimetype="application/json",
    )
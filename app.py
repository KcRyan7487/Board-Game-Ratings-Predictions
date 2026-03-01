from pathlib import Path

from flask import Flask, jsonify, render_template, request

from inference_utils import build_ordered_input_df, load_artifacts


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "artifacts" / "model.pkl"
FEATURES_PATH = APP_DIR / "artifacts" / "feature_columns.json"

model, feature_columns = load_artifacts(MODEL_PATH, FEATURES_PATH)

app = Flask(__name__)


FEATURE_OVERRIDES = {
    "GameWeight": {
        "label": "Game Weight (Complexity)",
        "hint": "BGG complexity score. Typical range is 1.0 to 5.0.",
        "min": 1.0,
        "max": 5.0,
        "step": "0.0001",
        "is_binary": False,
    },
    "YearPublished": {
        "label": "Year Published",
        "hint": "Publication year for the game.",
        "min": -5000, #There ARE some extremely old games in here with negatives in the thousands
        "max": 2030, #Should be a reasonable upper bound for now
        "step": "1",
        "is_binary": False,
    },
    "Cat_War": {
        "label": "War Category",
        "hint": "1 = Yes, categorized as war game. 0 = No.",
        "min": 0,
        "max": 1,
        "step": "1",
        "is_binary": True,
    },
    "mech_roll_spin_and_move": {
        "label": "Roll/Spin and Move Mechanic",
        "hint": "1 = Mechanic present. 0 = Mechanic not present.",
        "min": 0,
        "max": 1,
        "step": "1",
        "is_binary": True,
    },
}


def feature_meta(col: str) -> dict:
    override = FEATURE_OVERRIDES.get(col, {})

    is_binary = override.get("is_binary", col.startswith("mech_") or col.startswith("Cat_"))
    label = override.get("label", col.replace("_", " "))
    hint = override.get("hint", "Binary input: 0 = No, 1 = Yes" if is_binary else "")

    meta = {
        "name": col,
        "label": label,
        "hint": hint,
        "is_binary": is_binary,
        "step": override.get("step", "1" if is_binary or "Year" in col else "0.0001"),
        "min": override.get("min"),
        "max": override.get("max"),
    }
    return meta


FEATURES_META = [feature_meta(c) for c in feature_columns]
FEATURE_META_BY_NAME = {m["name"]: m for m in FEATURES_META}


def parse_and_validate_value(name: str, raw: str, meta: dict) -> float:
    label = meta["label"]
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"{label}: value is required.")

    try:
        val = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label}: must be a numeric value.") from exc

    low = meta.get("min")
    high = meta.get("max")
    if low is not None and val < float(low):
        raise ValueError(f"{label}: must be between {low} and {high}.")
    if high is not None and val > float(high):
        raise ValueError(f"{label}: must be between {low} and {high}.")

    if meta.get("is_binary"):
        if val not in (0.0, 1.0):
            raise ValueError(f"{label}: only 0 or 1 is allowed.")
        return int(val)

    # Keep year-like fields as int for cleaner display, otherwise keep float
    if meta.get("step") == "1":
        return int(round(val))
    return val


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        error=None,
        values={},
        feature_meta=FEATURES_META,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = {}
        errors = []
        for col in feature_columns:
            raw = request.form.get(col, "")
            meta = FEATURE_META_BY_NAME[col]
            try:
                payload[col] = parse_and_validate_value(col, raw, meta)
            except ValueError as exc:
                errors.append(str(exc))
                payload[col] = raw

        if errors:
            return render_template(
                "index.html",
                prediction=None,
                error="Please correct the following input issues: " + " | ".join(errors),
                values=payload,
                feature_meta=FEATURES_META,
            )

        input_df = build_ordered_input_df(payload, feature_columns)
        pred = float(model.predict(input_df)[0])

        return render_template(
            "index.html",
            prediction=round(pred, 6),
            error=None,
            values=payload,
            feature_meta=FEATURES_META,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            prediction=None,
            error=f"Prediction error: {exc}",
            values=request.form.to_dict(),
            feature_meta=FEATURES_META,
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "feature_bounds": {
                m["name"]: {"min": m.get("min"), "max": m.get("max")} for m in FEATURES_META
            },
        }
    )


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

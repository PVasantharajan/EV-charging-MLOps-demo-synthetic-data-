import pandas as pd
import joblib

from train import prepare_dataset


def predict(csv_path):

    clf = joblib.load("models/plug_model.pkl")
    reg = joblib.load("models/energy_model.pkl")

    df = prepare_dataset(csv_path)

    feature_cols = [
        "soc",
        "soc_delta",
        "hour",
        "day_of_week",
        "soc_lag_1",
        "soc_lag_2",
        "soc_lag_3",
    ]

    X = df[feature_cols]

    df["plug_probability"] = clf.predict_proba(X)[:, 1]
    df["predicted_energy_to_full"] = reg.predict(X)

    df[[
        "asset_id",
        "timestamp",
        "plug_probability",
        "predicted_energy_to_full",
    ]].to_csv("predictions.csv", index=False)

    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    predict("synthetic_ev_data.csv")

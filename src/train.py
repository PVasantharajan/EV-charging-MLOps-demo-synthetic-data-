import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor

from src.preprocessing import (
    parse_and_sort,
    compute_time_diff,
    flag_gaps,
    filter_short_assets,
)
from src.feature_engineering import (
    compute_soc_delta,
    add_time_features,
    add_lag_features,
    create_targets,
)


def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df = parse_and_sort(df)
    df = compute_time_diff(df)
    df = flag_gaps(df)
    df = filter_short_assets(df)

    df = compute_soc_delta(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = create_targets(df)

    df = df.dropna()

    return df


def train_models(csv_path):

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

    # -----------------------
    # Classification Model
    # -----------------------
    y_class = df["plug_future"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    clf.fit(X_train_c, y_train_c)

    prob_preds = clf.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, prob_preds)

    # -----------------------
    # Regression Model
    # -----------------------
    y_reg = df["energy_to_full_future"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    reg.fit(X_train_r, y_train_r)

    reg_preds = reg.predict(X_test_r)
    mae = mean_absolute_error(y_test_r, reg_preds)

    # -----------------------
    # Save artifacts
    # -----------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(clf, "models/plug_model.pkl")
    joblib.dump(reg, "models/energy_model.pkl")

    metrics = {
        "plug_model_auc": auc,
        "energy_model_mae": mae,
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Training complete")
    print("Plug model AUC:", auc)
    print("Energy model MAE:", mae)


if __name__ == "__main__":
    train_models("synthetic_ev_data.csv")

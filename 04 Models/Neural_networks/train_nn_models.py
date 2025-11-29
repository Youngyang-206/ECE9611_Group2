"""Train MLP and LSTM baselines for 6-hour consumption forecasting."""

from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT.parent / "01 Dataset" / "03 Data for model"

TRAIN_PATH = DATA_DIR / "Train_set.csv"
TEST_PATH = DATA_DIR / "Test_set.csv"

SEQ_LEN = 8
ERROR_DIR = Path(__file__).resolve().parent


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def prepare_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    drop_cols = ["household_ID", "DATE", "TIME", "timestamp", target_col]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        train_cat = X_train[col].astype("category")
        categories = train_cat.cat.categories
        X_train[col] = train_cat.cat.codes
        X_test[col] = pd.Categorical(X_test[col], categories=categories).codes

    for col in X_train.columns:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_test[col] = X_test[col].fillna(mean_val)

    return X_train, X_test, feature_cols


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    return X_train_scaled, X_test_scaled, scaler


def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def evaluate_and_save_errors(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> Tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    abs_error = np.abs(y_pred - y_true)
    squared_error = (y_pred - y_true) ** 2
    percentage_error = abs_error / (np.abs(y_true) + 1e-6)

    df_err = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": abs_error,
            "squared_error": squared_error,
            "percentage_error": percentage_error,
        }
    )
    out_path = ERROR_DIR / f"{model_name}_test_errors.csv"
    df_err.to_csv(out_path, index=False)
    print(
        f"[{model_name}] MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f} -> {out_path.name}"
    )
    return mae, rmse, r2


def attach_scaled_features(
    base_df: pd.DataFrame,
    scaled_features: np.ndarray,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    scaled_df = pd.DataFrame(
        scaled_features, columns=feature_cols, index=base_df.index
    )
    merged = pd.concat(
        [
            base_df[["household_ID", "timestamp", "future_6h_consumption"]].copy(),
            scaled_df,
        ],
        axis=1,
    )
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return merged


def build_sequences(
    df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for _, grp in df.groupby("household_ID"):
        grp = grp.sort_values("timestamp")
        feat_values = grp[feature_cols].values
        targets = grp["future_6h_consumption"].values

        if len(grp) < seq_len:
            continue

        for idx in range(seq_len - 1, len(grp)):
            window = feat_values[idx - seq_len + 1 : idx + 1]
            X_list.append(window)
            y_list.append(targets[idx])

    if not X_list:
        raise ValueError("No sequences were created; reduce seq_len or check data.")

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


def build_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    tf.random.set_seed(42)
    np.random.seed(42)

    train_df, test_df = load_datasets()
    target_col = "future_6h_consumption"

    X_train_df, X_test_df, feature_cols = prepare_feature_frames(
        train_df, test_df, target_col
    )
    X_train_scaled, X_test_scaled, _ = scale_features(X_train_df, X_test_df)

    mlp_model = build_mlp(X_train_scaled.shape[1])
    mlp_model.fit(
        X_train_scaled,
        train_df[target_col].values,
        epochs=30,
        batch_size=512,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=1,
    )
    y_pred_mlp = mlp_model.predict(X_test_scaled).flatten()
    evaluate_and_save_errors(test_df[target_col].values, y_pred_mlp, "MLP")
    mlp_model.save(ERROR_DIR / "MLP_model.h5")

    train_seq_df = attach_scaled_features(train_df, X_train_scaled, feature_cols)
    test_seq_df = attach_scaled_features(test_df, X_test_scaled, feature_cols)
    X_train_seq, y_train_seq = build_sequences(train_seq_df, feature_cols, SEQ_LEN)
    X_test_seq, y_test_seq = build_sequences(test_seq_df, feature_cols, SEQ_LEN)

    lstm_model = build_lstm((SEQ_LEN, len(feature_cols)))
    lstm_model.fit(
        X_train_seq,
        y_train_seq,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=1,
    )
    y_pred_lstm = lstm_model.predict(X_test_seq).flatten()
    evaluate_and_save_errors(y_test_seq, y_pred_lstm, "LSTM")
    lstm_model.save(ERROR_DIR / "LSTM_model.h5")


 

if __name__ == "__main__":
    main()

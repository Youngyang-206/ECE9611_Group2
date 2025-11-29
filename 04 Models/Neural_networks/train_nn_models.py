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
    """Load train and test datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def prepare_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict]:
    """
    Prepare features and handle encoding.
    Returns fitted preprocessing objects to avoid data leakage.
    """
    drop_cols = ["household_ID", "DATE", "TIME", "timestamp", target_col]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Store preprocessing info
    preprocessing_info = {
        'imputation_values': {},
        'category_mappings': {}
    }

    # Handle boolean columns
    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    # Handle categorical columns - FIT ON TRAIN ONLY
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        # Learn categories from training data only
        train_cat = X_train[col].astype("category")
        categories = train_cat.cat.categories
        preprocessing_info['category_mappings'][col] = categories
        
        # Encode training data
        X_train[col] = train_cat.cat.codes
        
        # Encode test data (unseen categories become -1)
        test_cat = pd.Categorical(X_test[col], categories=categories)
        X_test[col] = test_cat.codes
        
        # Handle unseen categories in test set
        unseen_mask = X_test[col] == -1
        if unseen_mask.any():
            # Set unseen categories to most frequent training category
            most_frequent = X_train[col].mode()[0]
            X_test.loc[unseen_mask, col] = most_frequent
            print(f"Warning: {unseen_mask.sum()} unseen categories in '{col}' "
                  f"set to most frequent value ({most_frequent})")

    # Handle missing values - LEARN FROM TRAIN ONLY
    for col in X_train.columns:
        mean_val = X_train[col].mean()
        preprocessing_info['imputation_values'][col] = mean_val
        X_train[col] = X_train[col].fillna(mean_val)
        X_test[col] = X_test[col].fillna(mean_val)

    return X_train, X_test, feature_cols, preprocessing_info


def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    FIT on training data only, TRANSFORM both train and test.
    """
    scaler = StandardScaler()
    # FIT on training data only
    X_train_scaled = scaler.fit_transform(X_train.values)
    # TRANSFORM test data using training statistics
    X_test_scaled = scaler.transform(X_test.values)
    return X_train_scaled, X_test_scaled, scaler


def build_mlp(input_dim: int) -> keras.Model:
    """Build Multi-Layer Perceptron model."""
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
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str
) -> Tuple[float, float, float]:
    """Evaluate model and save error metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
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
    print(f"\n{'='*60}")
    print(f"[{model_name}] Test Set Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Saved to: {out_path.name}")
    print(f"{'='*60}\n")
    return mae, rmse, r2


def attach_scaled_features(
    base_df: pd.DataFrame,
    scaled_features: np.ndarray,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Attach scaled features back to dataframe for sequence building."""
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
    df: pd.DataFrame, 
    feature_cols: Sequence[str], 
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sequences for LSTM from dataframe."""
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for household_id, grp in df.groupby("household_ID"):
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
        raise ValueError(
            f"No sequences were created. Reduce seq_len (current: {seq_len}) "
            "or check that households have enough data."
        )

    X = np.stack(X_list)
    y = np.array(y_list)
    print(f"Created {len(X)} sequences of length {seq_len}")
    return X, y


def build_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    """Build LSTM model for sequence prediction."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    """Main training pipeline."""
    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    print("="*60)
    print("MACHINE LEARNING PIPELINE - NO DATA LEAKAGE")
    print("="*60)

    # Load data
    print("\n1. Loading datasets...")
    train_df, test_df = load_datasets()
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    target_col = "future_6h_consumption"

    # Prepare features (fit preprocessing on train only)
    print("\n2. Preparing features (FIT on train, TRANSFORM on test)...")
    X_train_df, X_test_df, feature_cols, preprocessing_info = prepare_feature_frames(
        train_df, test_df, target_col
    )
    print(f"   Features: {len(feature_cols)}")

    # Scale features (fit scaler on train only)
    print("\n3. Scaling features (FIT on train, TRANSFORM on test)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_df, X_test_df)

    # ===== MLP MODEL =====
    print("\n" + "="*60)
    print("TRAINING MLP MODEL")
    print("="*60)
    
    mlp_model = build_mlp(X_train_scaled.shape[1])
    print(f"\nModel architecture:")
    mlp_model.summary()
    
    print("\nTraining MLP...")
    mlp_history = mlp_model.fit(
        X_train_scaled,
        train_df[target_col].values,
        epochs=30,
        batch_size=512,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=3, 
                restore_best_weights=True,
                verbose=1
            )
        ],
        verbose=1,
    )
    
    print("\nEvaluating MLP on TEST set...")
    y_pred_mlp = mlp_model.predict(X_test_scaled, verbose=0).flatten()
    evaluate_and_save_errors(test_df[target_col].values, y_pred_mlp, "MLP")
    
    mlp_save_path = ERROR_DIR / "MLP_model.keras"
    mlp_model.save(mlp_save_path)
    print(f"MLP model saved to: {mlp_save_path}")

    # ===== LSTM MODEL =====
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    print("\n4. Building sequences for LSTM...")
    train_seq_df = attach_scaled_features(train_df, X_train_scaled, feature_cols)
    test_seq_df = attach_scaled_features(test_df, X_test_scaled, feature_cols)
    
    print("   Training sequences...")
    X_train_seq, y_train_seq = build_sequences(train_seq_df, feature_cols, SEQ_LEN)
    print("   Test sequences...")
    X_test_seq, y_test_seq = build_sequences(test_seq_df, feature_cols, SEQ_LEN)

    lstm_model = build_lstm((SEQ_LEN, len(feature_cols)))
    print(f"\nModel architecture:")
    lstm_model.summary()
    
    print("\nTraining LSTM...")
    lstm_history = lstm_model.fit(
        X_train_seq,
        y_train_seq,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=3, 
                restore_best_weights=True,
                verbose=1
            )
        ],
        verbose=1,
    )
    
    print("\nEvaluating LSTM on TEST set...")
    y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
    evaluate_and_save_errors(y_test_seq, y_pred_lstm, "LSTM")
    
    lstm_save_path = ERROR_DIR / "LSTM_model.keras"
    lstm_model.save(lstm_save_path)
    print(f"LSTM model saved to: {lstm_save_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
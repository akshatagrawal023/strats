"""
train_vol_model.py
------------------
Trains an XGBoost classifier to identify moments where selling
implied volatility (via Iron Condor/Butterfly) has a statistical edge.

The model learns to recognize vol surface "signatures" that precede
IV overpricing relative to realized vol — pure variance risk premium harvesting.
"""

import os
import sys
import glob
import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper_trading.ml.vol_data_prep import load_vol_arb_dataset


def find_all_h5(folder: str = "../hdf5_data_archives") -> list:
    """Find all HDF5 files in the archive directory."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
    files = sorted(glob.glob(os.path.join(path, "*.h5")))
    return files


def train_vol_arb_model():
    print("=" * 60)
    print("  VOLATILITY ARBITRAGE MODEL TRAINING")
    print("  Strategy: Iron Condor / Iron Butterfly Edge Detection")
    print("=" * 60)
    
    h5_files = find_all_h5()
    if not h5_files:
        print("\nNo HDF5 data found in hdf5_data_archives/.")
        print("Run the paper trading orchestrator to collect data first.")
        return
    
    print(f"\nFound {len(h5_files)} data file(s):")
    for f in h5_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load all files and concatenate
    all_X, all_y = [], []
    for filepath in h5_files:
        X, y = load_vol_arb_dataset(
            filepath,
            lookahead_ticks=600,   # 30 min @ 3s/tick
            rv_window=120          # 6 min lookback for realized vol
        )
        if not X.empty:
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        print("No valid data extracted. Need more trading data.")
        return
    
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    
    print(f"\n--- Combined Dataset ---")
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Edge exists (Y=1): {y.mean():.1%}")
    
    # Temporal split: train on first 80%, test on last 20%
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Handle class imbalance
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    
    if n_pos == 0:
        print("\nNo positive targets (Y=1) found.")
        print("This means IV was never overpriced by >2 vol points in the training window.")
        print("Try lowering the edge threshold or collecting more data.")
        return
    
    if n_neg == 0:
        print("\nAll targets are positive (Y=1).")
        print("IV was always overpriced — edge existed everywhere. Model cannot learn discriminatively.")
        return
    
    scale_weight = n_neg / n_pos
    
    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Class weight: {scale_weight:.2f}")
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        early_stopping_rounds=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=20
    )
    
    # --- Evaluation ---
    print("\n" + "=" * 60)
    print("  EVALUATION ON HOLD-OUT TEST SET")
    print("=" * 60)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, preds))
    
    try:
        auc = roc_auc_score(y_test, probs)
        print(f"ROC AUC Score: {auc:.4f}")
    except ValueError:
        print("ROC AUC not computable (single class in test set)")
    
    # High-confidence signals
    for threshold in [0.70, 0.80, 0.90]:
        hc_preds = (probs > threshold).astype(int)
        n_signals = np.sum(hc_preds)
        if n_signals > 0:
            prec = precision_score(y_test, hc_preds, zero_division=0)
            print(f"  Threshold {threshold:.0%}: {n_signals} signals, Win Rate: {prec:.1%}")
        else:
            print(f"  Threshold {threshold:.0%}: 0 signals")
    
    # --- Feature Importance ---
    fig, ax = plt.subplots(figsize=(12, 10))
    xgb.plot_importance(
        model, max_num_features=20, ax=ax,
        title="Top 20 Features: Vol Arb Edge Detection",
        importance_type='gain'
    )
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "vol_feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved feature importance plot to {plot_path}")
    
    # --- Save Model ---
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "vol_arb_v1.pkl")
    with open(weight_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model weights to {weight_path}")
    
    # Save feature names for inference
    feature_names_path = os.path.join(weights_dir, "vol_arb_v1_features.pkl")
    with open(feature_names_path, "wb") as f:
        pickle.dump(list(X.columns), f)
    print(f"Saved feature names to {feature_names_path}")


if __name__ == "__main__":
    import pandas as pd
    train_vol_arb_model()

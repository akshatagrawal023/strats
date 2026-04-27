import os
import glob
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, accuracy_score
import matplotlib.pyplot as plt
from paper_trading.ml.data_prep import load_and_prep_h5

def find_latest_h5(folder: str = "../hdf5_data_archives") -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
    files = glob.glob(os.path.join(path, "*.h5"))
    if not files:
        return ""
    return max(files, key=os.path.getctime)

def train_bullish_entry_model():
    print("=== Training Bullish Entry XGBoost Model ===")
    
    h5_file = find_latest_h5()
    if not h5_file:
        print("No HDF5 data found. Run the paper trading pipeline first.")
        return
        
    X, y = load_and_prep_h5(h5_file, lookahead_ticks=600, target_profit_bps=10.0, stop_loss_bps=10.0)
    
    if X.empty:
        print("Not enough data to train. Pipeline needs to run longer.")
        return
        
    # Temporal Train/Test split is critical in finance. Don't use random shuffle!
    # We use the first 80% of the day to train, and the last 20% to test.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Calculate scale_pos_weight to handle class imbalance
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    if num_pos == 0:
        print("No positive targets found in training set. Adjust profit/stop loss levels.")
        return
    
    scale_weight = num_neg / num_pos
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Class imbalance weight applied: {scale_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10
    )
    
    # Evaluation
    print("\n--- Evaluate against hold-out Test Set ---")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, preds))
    
    # Let's say we only execute when the model is 85% confident
    high_conf_preds = (probs > 0.85).astype(int)
    print("\n--- High Confidence Precision (>85% Prob) ---")
    print(f"Number of signals triggered: {np.sum(high_conf_preds)}")
    if np.sum(high_conf_preds) > 0:
        prec = precision_score(y_test, high_conf_preds, zero_division=0)
        print(f"Win Rate (Precision) of these trades: {prec:.2%}")
        
    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, ax=ax, title="Top 15 Features for Bullish Breaks")
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "feature_importance.png")
    plt.savefig(plot_path)
    print(f"Saved feature importance plot to {plot_path}")
    
    # Save the model
    os.makedirs(os.path.join(os.path.dirname(__file__), "weights"), exist_ok=True)
    weight_path = os.path.join(os.path.dirname(__file__), "weights", "bullish_entry_v1.pkl")
    with open(weight_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model weights to {weight_path}")

if __name__ == "__main__":
    import numpy as np
    train_bullish_entry_model()

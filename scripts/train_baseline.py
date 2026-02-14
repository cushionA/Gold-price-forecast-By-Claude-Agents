"""
Train XGBoost baseline model for Phase 1
Split data (70/15/15), train model, evaluate, and save results
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

# Paths
DATA_PROCESSED = Path("data/processed")
SHARED = Path("shared")
MODELS = Path("models")

def calculate_sharpe_ratio(y_true, y_pred, transaction_cost_bps=5):
    """
    Calculate Sharpe ratio with transaction costs
    Transaction cost: 5bps per trade (one-way)
    """
    # Generate trading signals from predictions
    signals = np.sign(y_pred)

    # Calculate raw returns based on signals
    raw_returns = signals * y_true / 100  # Convert % to decimal

    # Calculate transaction costs
    # Trade when signal changes (including first position)
    trades = np.abs(np.diff(signals, prepend=0))
    costs = trades * (transaction_cost_bps / 10000)  # bps to decimal

    # Net returns after costs
    net_returns = raw_returns - costs

    # Calculate Sharpe ratio (annualized)
    if net_returns.std() == 0:
        return 0.0

    sharpe = np.sqrt(252) * net_returns.mean() / net_returns.std()
    return sharpe

def calculate_direction_accuracy(y_true, y_pred):
    """
    Calculate direction accuracy
    Exclude exactly zero returns from calculation
    """
    # Filter out zero returns
    nonzero_mask = y_true != 0
    y_true_filtered = y_true[nonzero_mask]
    y_pred_filtered = y_pred[nonzero_mask]

    if len(y_true_filtered) == 0:
        return 0.0

    # Calculate accuracy
    correct = np.sum(np.sign(y_true_filtered) == np.sign(y_pred_filtered))
    accuracy = correct / len(y_true_filtered)

    return accuracy

def main():
    print("Loading base features...")
    df = pd.read_csv(DATA_PROCESSED / "base_features.csv", index_col=0, parse_dates=True)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Split features and target
    target_col = 'gold_return_next'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {len(X)}")

    # Time-series split (70/15/15)
    n_total = len(X)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train:n_train+n_val]
    y_val = y.iloc[n_train:n_train+n_val]

    X_test = X.iloc[n_train+n_val:]
    y_test = y.iloc[n_train+n_val:]

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
    print(f"  Val:   {len(X_val)} samples ({X_val.index.min()} to {X_val.index.max()})")
    print(f"  Test:  {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")

    # Train XGBoost model
    print("\nTraining XGBoost baseline...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Evaluate on all sets
    metrics = {}

    for split_name, y_true, y_pred in [
        ('train', y_train, y_train_pred),
        ('val', y_val, y_val_pred),
        ('test', y_test, y_test_pred)
    ]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        direction_acc = calculate_direction_accuracy(y_true.values, y_pred)
        sharpe = calculate_sharpe_ratio(y_true.values, y_pred)

        metrics[split_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_acc),
            'sharpe_ratio': float(sharpe),
            'n_samples': int(len(y_true))
        }

        print(f"\n{split_name.upper()} Set Metrics:")
        print(f"  MAE:                {mae:.4f}%")
        print(f"  RMSE:               {rmse:.4f}%")
        print(f"  Direction Accuracy: {direction_acc:.2%}")
        print(f"  Sharpe Ratio:       {sharpe:.3f}")

    # Calculate high-confidence accuracy (top 30% predictions by magnitude)
    test_pred_abs = np.abs(y_test_pred)
    threshold = np.percentile(test_pred_abs, 70)
    high_conf_mask = test_pred_abs >= threshold

    if high_conf_mask.sum() > 0:
        hc_accuracy = calculate_direction_accuracy(
            y_test.values[high_conf_mask],
            y_test_pred[high_conf_mask]
        )
        metrics['test']['high_confidence_direction_accuracy'] = float(hc_accuracy)
        print(f"  High-Conf DA:       {hc_accuracy:.2%} (n={high_conf_mask.sum()})")

    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop 10 Features by Importance:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")

    # Save baseline results
    baseline_score = {
        'model': 'XGBoost',
        'created_at': datetime.now().isoformat(),
        'data_split': {
            'train': n_train,
            'val': n_val,
            'test': len(X_test),
            'total': n_total
        },
        'metrics': metrics,
        'feature_importance': {feat: float(imp) for feat, imp in top_features},
        'hyperparameters': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate,
            'subsample': model.subsample,
            'colsample_bytree': model.colsample_bytree
        },
        'targets': {
            'direction_accuracy': 0.56,
            'high_confidence_direction_accuracy': 0.60,
            'mae': 0.75,
            'sharpe_ratio': 0.8
        }
    }

    baseline_path = SHARED / "baseline_score.json"
    with open(baseline_path, 'w') as f:
        json.dump(baseline_score, f, indent=2)
    print(f"\nSaved baseline score to {baseline_path}")

    # Save model
    model_dir = MODELS / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_baseline.ubj"
    model.get_booster().save_model(model_path)
    print(f"Saved model to {model_path}")

    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        'date': X_test.index,
        'actual': y_test.values,
        'predicted': y_test_pred
    })
    pred_path = model_dir / "test_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved test predictions to {pred_path}")

    print("\n=== Baseline Model Summary ===")
    print(f"Test MAE:  {metrics['test']['mae']:.4f}% (target: <0.75%)")
    print(f"Test DA:   {metrics['test']['direction_accuracy']:.2%} (target: >56%)")
    print(f"Test HC-DA: {metrics['test'].get('high_confidence_direction_accuracy', 0):.2%} (target: >60%)")
    print(f"Test Sharpe: {metrics['test']['sharpe_ratio']:.3f} (target: >0.8)")

    print("\nBaseline training complete!")

if __name__ == "__main__":
    main()

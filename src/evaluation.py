"""
Evaluation Metrics - Used by evaluator and in training scripts
"""

import numpy as np
import pandas as pd
from scipy.stats import mutual_info_regression
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Tuple, Optional


def compute_overfit_ratio(train_loss: float, val_loss: float) -> float:
    """
    Compute overfit ratio (Gate 1 check)

    Args:
        train_loss: Training loss
        val_loss: Validation loss

    Returns:
        Overfit ratio (val_loss / train_loss)
    """
    if train_loss == 0:
        return float('inf')
    return val_loss / train_loss


def check_output_quality(output: pd.DataFrame) -> Dict[str, bool]:
    """
    Check basic output quality (Gate 1 checks)

    Returns:
        Dictionary with check results
    """
    checks = {}

    # Check for all NaN
    checks['no_all_nan'] = not output.isna().all().any()

    # Check for constant values
    checks['no_constant'] = (output.nunique() > 1).all()

    # Check for reasonable range (not all zeros, not all same value)
    checks['reasonable_range'] = (output.std() > 1e-6).all()

    return checks


def compute_mutual_information(submodel_output: pd.DataFrame,
                               target: pd.Series,
                               base_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Compute mutual information for Gate 2

    Args:
        submodel_output: Submodel output features
        target: Gold return target
        base_features: Baseline features (for comparison)

    Returns:
        Dictionary with MI metrics
    """
    # Align data
    common_index = submodel_output.index.intersection(target.index)
    sub_aligned = submodel_output.loc[common_index]
    target_aligned = target.loc[common_index]

    # Remove NaN
    valid_mask = ~(sub_aligned.isna().any(axis=1) | target_aligned.isna())
    sub_clean = sub_aligned[valid_mask]
    target_clean = target_aligned[valid_mask]

    if len(target_clean) < 10:
        return {'mi_total': 0.0, 'mi_gain_pct': 0.0}

    # Compute MI for submodel output
    mi_values = []
    for col in sub_clean.columns:
        mi = mutual_info_regression(
            sub_clean[[col]].values,
            target_clean.values,
            random_state=42
        )[0]
        mi_values.append(mi)

    mi_total = sum(mi_values)

    # Compute MI gain if base features provided
    mi_gain_pct = 0.0
    if base_features is not None:
        base_aligned = base_features.loc[common_index][valid_mask]
        mi_base_values = []
        for col in base_aligned.columns:
            mi = mutual_info_regression(
                base_aligned[[col]].values,
                target_clean.values,
                random_state=42
            )[0]
            mi_base_values.append(mi)

        mi_base_total = sum(mi_base_values)
        if mi_base_total > 0:
            mi_gain_pct = ((mi_total - mi_base_total) / mi_base_total) * 100

    return {
        'mi_total': mi_total,
        'mi_gain_pct': mi_gain_pct
    }


def compute_vif(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (Gate 2 check)

    Args:
        features: Feature DataFrame

    Returns:
        DataFrame with VIF for each feature
    """
    # Remove NaN
    clean_features = features.dropna()

    if len(clean_features) < 10:
        return pd.DataFrame({'feature': features.columns, 'VIF': [np.nan] * len(features.columns)})

    vif_data = []
    for i, col in enumerate(clean_features.columns):
        try:
            vif = variance_inflation_factor(clean_features.values, i)
            vif_data.append({'feature': col, 'VIF': vif})
        except:
            vif_data.append({'feature': col, 'VIF': np.nan})

    return pd.DataFrame(vif_data)


def compute_rolling_correlation(submodel_output: pd.DataFrame,
                                base_features: pd.DataFrame,
                                window: int = 60) -> Dict[str, float]:
    """
    Compute rolling correlation stability (Gate 2 check)

    Args:
        submodel_output: Submodel output
        base_features: Base features
        window: Rolling window size

    Returns:
        Dictionary with correlation metrics
    """
    common_cols = submodel_output.columns.intersection(base_features.columns)

    if len(common_cols) == 0:
        return {'corr_std_mean': 0.0, 'corr_std_max': 0.0}

    corr_stds = []
    for col in common_cols:
        rolling_corr = submodel_output[col].rolling(window).corr(base_features[col])
        corr_std = rolling_corr.std()
        if not np.isnan(corr_std):
            corr_stds.append(corr_std)

    if len(corr_stds) == 0:
        return {'corr_std_mean': 0.0, 'corr_std_max': 0.0}

    return {
        'corr_std_mean': np.mean(corr_stds),
        'corr_std_max': np.max(corr_stds)
    }


def compute_direction_accuracy(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Compute directional accuracy (Gate 3 / final metric)

    Args:
        predictions: Predicted returns
        actuals: Actual returns

    Returns:
        Direction accuracy (%)
    """
    # Align
    common_index = predictions.index.intersection(actuals.index)
    pred_aligned = predictions.loc[common_index]
    actual_aligned = actuals.loc[common_index]

    # Remove zeros (np.sign(0) = 0 issue)
    nonzero_mask = (actual_aligned != 0) & (pred_aligned != 0)
    pred_signs = np.sign(pred_aligned[nonzero_mask])
    actual_signs = np.sign(actual_aligned[nonzero_mask])

    if len(actual_signs) == 0:
        return 0.0

    accuracy = (pred_signs == actual_signs).sum() / len(actual_signs) * 100
    return accuracy


def compute_sharpe_ratio(returns: pd.Series, transaction_cost_bps: float = 5.0) -> float:
    """
    Compute Sharpe ratio with transaction costs (Gate 3 / final metric)

    Args:
        returns: Strategy returns (%)
        transaction_cost_bps: Transaction cost in basis points (one-way)

    Returns:
        Sharpe ratio
    """
    # Deduct transaction costs (assume trading on every signal)
    cost_pct = transaction_cost_bps / 100  # Convert bps to %
    net_returns = returns - cost_pct

    if len(net_returns) < 2 or net_returns.std() == 0:
        return 0.0

    # Annualized Sharpe (assuming daily returns)
    sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
    return sharpe


def evaluate_ablation(baseline_pred: pd.Series,
                     with_submodel_pred: pd.Series,
                     actuals: pd.Series) -> Dict[str, float]:
    """
    Evaluate ablation test (Gate 3)

    Args:
        baseline_pred: Baseline model predictions
        with_submodel_pred: Model with submodel predictions
        actuals: Actual returns

    Returns:
        Dictionary with ablation metrics
    """
    # Direction accuracy
    baseline_da = compute_direction_accuracy(baseline_pred, actuals)
    with_submodel_da = compute_direction_accuracy(with_submodel_pred, actuals)
    da_delta = with_submodel_da - baseline_da

    # MAE
    common_index = baseline_pred.index.intersection(actuals.index).intersection(with_submodel_pred.index)
    baseline_mae = mean_absolute_error(actuals.loc[common_index], baseline_pred.loc[common_index])
    with_submodel_mae = mean_absolute_error(actuals.loc[common_index], with_submodel_pred.loc[common_index])
    mae_delta = baseline_mae - with_submodel_mae  # Positive = improvement

    # Sharpe ratio
    baseline_sharpe = compute_sharpe_ratio(baseline_pred)
    with_submodel_sharpe = compute_sharpe_ratio(with_submodel_pred)
    sharpe_delta = with_submodel_sharpe - baseline_sharpe

    return {
        'baseline_direction_accuracy': baseline_da,
        'with_submodel_direction_accuracy': with_submodel_da,
        'direction_accuracy_delta': da_delta,
        'baseline_mae': baseline_mae,
        'with_submodel_mae': with_submodel_mae,
        'mae_delta': mae_delta,
        'baseline_sharpe': baseline_sharpe,
        'with_submodel_sharpe': with_submodel_sharpe,
        'sharpe_delta': sharpe_delta
    }


def check_gate_3_pass(ablation_results: Dict[str, float]) -> bool:
    """
    Check if Gate 3 criteria are met (any one condition)

    Args:
        ablation_results: Results from evaluate_ablation

    Returns:
        True if any Gate 3 condition is met
    """
    conditions = [
        ablation_results['direction_accuracy_delta'] >= 0.5,  # +0.5%
        ablation_results['sharpe_delta'] >= 0.05,              # +0.05
        ablation_results['mae_delta'] >= 0.01                  # -0.01% MAE (positive delta = improvement)
    ]

    return any(conditions)

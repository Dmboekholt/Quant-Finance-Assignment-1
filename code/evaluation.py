from arch import arch_model
import numpy as np
import pandas as pd
import logging
from scipy import stats

def evaluate_models(variance_proxy, df, horizon=1):
    """
    Generate rolling forecasts for GARCH models with specified horizon.
    
    Parameters:
    variance_proxy : array-like
        Variance proxy for evaluation period
    df : pd.DataFrame
        Full dataset with 'R' (returns) and 'RV' (realised variance) columns
    horizon : int
        Forecast horizon (default=1 for one-step-ahead)
    
    Returns:
    dict : {'GARCH': array, 'ARCH': array, 'Threshold_GARCH': array, 'GARCH_X': array}
    """
    garch_var_forecast = []
    arch_var_forecast = []
    threshold_garch_var_forecast = []
    garchx_var_forecast = []
    
    for i in range(len(variance_proxy)):
        current_idx = 1258 + i
        sample_for_refit = df['R'].iloc[:current_idx]
        
        # GARCH
        garch_model_refit = arch_model(sample_for_refit, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
        garch_result_refit = garch_model_refit.fit(disp='off')
        garch_f = garch_result_refit.forecast(horizon=horizon)
        garch_var_forecast.append(garch_f.variance.values[-1, horizon-1]) 
        
        # ARCH
        arch_model_refit = arch_model(sample_for_refit, mean='Constant', vol='ARCH', q=1, dist='normal')
        arch_result_refit = arch_model_refit.fit(disp='off')
        arch_f = arch_result_refit.forecast(horizon=horizon)
        arch_var_forecast.append(arch_f.variance.values[-1, horizon-1])
            
        # Threshold GARCH
        threshold_garch_model_refit = arch_model(sample_for_refit, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='normal')   
        threshold_garch_result_refit = threshold_garch_model_refit.fit(disp='off')
        threshold_garch_f = threshold_garch_result_refit.forecast(horizon=horizon)        
        threshold_garch_var_forecast.append(threshold_garch_f.variance.values[-1, horizon-1])
            
        # GARCH-X
        rv_sample_refit = df['RV'].iloc[:current_idx]
        exog_refit = rv_sample_refit.shift(1)
        data_refit = pd.DataFrame({'R': sample_for_refit, 'RV_lag1': exog_refit}).dropna()
        y_refit = data_refit['R']
        X_refit = data_refit['RV_lag1']
        garchx_model_refit = arch_model(y_refit, mean='Constant', vol='GARCH', p=1, q=1, x=X_refit, dist='normal')
        garchx_result_refit = garchx_model_refit.fit(disp='off')
        garchx_f = garchx_result_refit.forecast(horizon=horizon)
        garchx_var_forecast.append(garchx_f.variance.values[-1, horizon-1])

    results = {
        'GARCH': np.array(garch_var_forecast),
        'ARCH': np.array(arch_var_forecast),
        'Threshold_GARCH': np.array(threshold_garch_var_forecast),
        'GARCH_X': np.array(garchx_var_forecast)
    }

    return results

def calculate_MSPE(actual, predicted, model_name=None):
    """
    Calculate Mean Squared Prediction Error (MSPE)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    valid_mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual_clean = actual[valid_mask]
    predicted_clean = predicted[valid_mask]
    mspe = np.mean((actual_clean - predicted_clean) ** 2)
    
    if model_name:
        logging.info(f"A1.7.1 MSPE for {model_name}: {mspe:.6f}")

    return mspe

def calculate_var_estimates(returns, volatility_forecasts, confidence_levels=[0.90, 0.95, 0.99]):
    """
    Calculate 1-day ahead Value-at-Risk (VaR) estimates.
    
    Parameters:
    returns : array-like
        Daily returns (realized)
    volatility_forecasts : dict
        Dictionary with model names as keys and volatility forecasts as values
    confidence_levels : list
        List of confidence levels (e.g., [0.90, 0.95, 0.99])
    dict : VaR estimates for each model and confidence level
    """
    returns = np.array(returns)
    # Convert variance to volatility by taking square root
    volatility_forecasts = {k: np.sqrt(np.array(v)) for k, v in volatility_forecasts.items()}  
    
    var_estimates = {}

    for confidence in confidence_levels:
        var_estimates[confidence] = {}
        z_score = stats.norm.ppf(1 - confidence)  # Left tail (negative for losses)
        
        for model_name, volatility in volatility_forecasts.items():
            # VaR = mean + z_score * volatility (assuming mean ≈ 0 for daily returns)
            var = z_score * volatility
            var_estimates[confidence][model_name] = var
    
    return var_estimates


def calculate_var_violations(returns, var_estimates, confidence_levels=[0.90, 0.95, 0.99]):
    """
    Calculate VaR violations (backtesting) - count when returns breach VaR estimates.
    
    Parameters:
    returns : array-like
        Daily returns
    var_estimates : dict
        VaR estimates from calculate_var_estimates
    confidence_levels : list
        List of confidence levels
    
    Returns:
    dict : Violation counts and rates for each model and confidence level
    """
    returns = np.array(returns)
    violations = {}
    
    for confidence in confidence_levels:
        violations[confidence] = {}
        expected_violations = (1 - confidence) * len(returns)
        
        for model_name, var_forecast in var_estimates[confidence].items():
            var_forecast = np.array(var_forecast)
            # Count violations: where returns < VaR (losses exceed VaR)
            breaches = np.sum(returns < var_forecast)
            violation_rate = breaches / len(returns)
            expected_rate = 1 - confidence
            
            violations[confidence][model_name] = {
                'breaches': breaches,
                'expected_breaches': expected_violations,
                'violation_rate': violation_rate,
                'expected_rate': expected_rate
            }
    
    return violations

def christoffersen_uc_test(returns, var_estimates, confidence_level=0.95):
    """
    Christoffersen (1998) Unconditional Coverage (UC) test.
    Tests if the number of VaR violations equals the expected number.
    
    Parameters:
    returns : array-like
        Daily returns
    var_estimates : dict
        VaR estimates for a specific confidence level
    confidence_level : float
        Confidence level (e.g., 0.95)
    
    Returns:
    dict : Test statistics and p-values for each model
    """
    returns = np.array(returns)
    n = len(returns)
    p = 1 - confidence_level  
    
    results = {}
    
    for model_name, var_forecast in var_estimates.items():
        var_forecast = np.array(var_forecast)
        violations = np.sum(returns < var_forecast)
        
        # Likelihood Ratio test statistic
        # LR_UC = 2 * [ln(L(p)) - ln(L(p_hat))]
        # where p_hat = violations / n
        p_hat = violations / n
        
        if p_hat > 0 and p_hat < 1:
            lr_uc = 2 * (violations * np.log(p_hat / p) + (n - violations) * np.log((1 - p_hat) / (1 - p)))
        else:
            lr_uc = np.nan
        
        # p-value from chi-squared distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_uc, df=1)
        
        results[model_name] = {
            'violations': violations,
            'expected_violations': n * p,
            'violation_rate': p_hat,
            'expected_rate': p,
            'LR_UC': lr_uc,
            'p_value': p_value,
            'reject_h0': p_value < 0.05  # Reject if p < 0.05
        }
    
    return results


def christoffersen_ind_test(returns, var_estimates, confidence_level=0.95):
    """
    Christoffersen (1998) Independence (Ind) test.
    Tests if VaR violations are independently distributed (no clustering).
    
    Parameters:
    returns : array-like
        Daily returns
    var_estimates : dict
        VaR estimates for a specific confidence level
    confidence_level : float
        Confidence level (e.g., 0.95)
    
    Returns:
    dict : Test statistics and p-values for each model
    """
    returns = np.array(returns)
    n = len(returns)
    p = 1 - confidence_level
    
    results = {}
    
    for model_name, var_forecast in var_estimates.items():
        var_forecast = np.array(var_forecast)
        violations = (returns < var_forecast).astype(int)
        
        # Count transitions: 00, 01, 10, 11
        n_00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n_01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n_10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n_11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
        
        den1 = n_00 + n_01  # Total transitions from state 0
        den2 = n_10 + n_11  # Total transitions from state 1
        total = den1 + den2
        
        # If we cannot estimate transition probs (no 0-states or no 1-states), IND test is undefined
        if den1 == 0 or den2 == 0 or total == 0:
            lr_ind = np.nan
            p_value = np.nan
        else:
            # Transition probabilities
            p11 = n_01 / den1  # P(I_{t}=1 | I_{t-1}=0)
            p12 = n_11 / den2  # P(I_{t}=1 | I_{t-1}=1)
            pi = (n_01 + n_11) / total  # Unconditional violation prob
            
            # Clip probabilities to avoid log(0)
            eps = 1e-12
            p11 = np.clip(p11, eps, 1 - eps)
            p12 = np.clip(p12, eps, 1 - eps)
            pi = np.clip(pi, eps, 1 - eps)
            
            # Log-likelihood under independence (L_0)
            log_l0 = ((n_00 + n_10) * np.log(1 - pi) + 
                      (n_01 + n_11) * np.log(pi))
            
            # Log-likelihood under dependence (L_1)
            log_l1 = (n_00 * np.log(1 - p11) + n_01 * np.log(p11) +
                      n_10 * np.log(1 - p12) + n_11 * np.log(p12))
            
            lr_ind = 2 * (log_l1 - log_l0)
            p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
        
        results[model_name] = {
            'transitions_00': n_00,
            'transitions_01': n_01,
            'transitions_10': n_10,
            'transitions_11': n_11,
            'total_violations': n_01 + n_11,
            'LR_Ind': lr_ind,
            'p_value': p_value,
            'reject_h0': p_value < 0.05 if not np.isnan(p_value) else np.nan
        }
    
    return results

from arch import arch_model
import numpy as np
import pandas as pd
import logging

def evaluate_models(variance_proxy, df, horizon=1):
    """
    Generate rolling forecasts for GARCH models with specified horizon.
    
    Parameters:
    -----------
    variance_proxy : array-like
        Variance proxy for evaluation period
    df : pd.DataFrame
        Full dataset with 'R' (returns) and 'RV' (realised variance) columns
    horizon : int
        Forecast horizon (default=1 for one-step-ahead)
    
    Returns:
    --------
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

def output_evaluation_results(variance_proxy, forecasts, horizon=1):
    """
    Calculate and log MSPE for each model's forecasts.
    
    Parameters:
    -----------
    variance_proxy : array-like
        Variance proxy for evaluation period
    forecasts : dict
        Dictionary containing model forecasts with keys: 'GARCH', 'ARCH', 'Threshold_GARCH', 'GARCH_X'
    horizon : int
        Forecast horizon used (for logging purposes, default=1)
    """
    for model_name, predicted in forecasts.items():
        mspe = calculate_MSPE(variance_proxy, predicted, model_name=model_name)
        logging.info(f"Model: {model_name}, MSPE: {mspe:.6f}, Horizon: {horizon}")
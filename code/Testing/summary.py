from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import kurtosis, skew, jarque_bera
import numpy as np
import logging

def summary_normality_assessment(data, nlags=20):
    """
    Calculate summary statistics for a given dataset
    """
    mean = data.mean()
    std = data.std()
    kurtosis_value = kurtosis(data)
    skew_value = skew(data)
    jarque_bera_test_value = jarque_bera(data)
    acf_values = acf(data, nlags=nlags)
    pacf_values = pacf(data, nlags=nlags)

    return mean, std, kurtosis_value, skew_value, jarque_bera_test_value, acf_values, pacf_values   

def summary_autocorrelation_assessment(data, nlags=20):
    """
    Calculate summary statistics for a given dataset
    """
    acf_values = acf(data, nlags=nlags)
    pacf_values = pacf(data, nlags=nlags)

    return acf_values, pacf_values


def summary_basic_statistics(data):
    """
    Calculate summary statistics for a given dataset
    """
    mean = data.mean()
    std = data.std()

    return mean, std

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


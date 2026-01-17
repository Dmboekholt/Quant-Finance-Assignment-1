from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import kurtosis, skew, jarque_bera

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
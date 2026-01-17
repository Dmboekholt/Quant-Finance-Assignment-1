"""
Machine Learning Assignment
Main script for data processing and model training
"""

# Basic Imports
import os
from re import L
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Import configuration
from config import DATA_DIR, PLOTS_DIR, setup_logging

# Import logging
import logging

# Import summary statistics
from Testing.summary import summary_normality_assessment, summary_autocorrelation_assessment, summary_basic_statistics
from Plotting.plots import plot_histogram, plot_acf, plot_pacf, plot_timeseries, plot_multiple_timeseries
from Logging.normality import log_normality_assessment

def main():
    """
    Main function for machine learning assignment
    """
    
    # Get data folder path from config
    data_folder = DATA_DIR
    df = pd.read_excel(os.path.join(data_folder, 'JNJ1423.xlsx'))
    
    # Convert DATE column to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Log initial returns data
    logging.info("Initial Returns Data (R - continuously compounded daily returns in percentage points):")
    logging.info(f"  Number of observations: {len(df['R'])}")
    logging.info(f"  Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    logging.info(f"  Statistics:")
    logging.info(f"    Min: {df['R'].min():.6f}")
    logging.info(f"    Max: {df['R'].max():.6f}")
    logging.info(f"    Mean: {df['R'].mean():.6f}")
    logging.info(f"    Std: {df['R'].std():.6f}")



    #----------------------------------------#
    # A1.1 Stylized Facts of Returns and Realised Variance 
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.1 Stylized Facts of Returns and Realised Variance \n----------------------------------------")

    # A1.1.1 Asses Normality of Returns
    mean_returns, std_returns, kurtosis_returns, skew_returns, \
        jarque_bera_test, _, _ = summary_normality_assessment(df['R'])
    log_normality_assessment(mean_returns, std_returns, kurtosis_returns, skew_returns, 
                             jarque_bera_test, prefix="A1.1.1", variable_name="Returns")
    plot_histogram(df['R'], 'Histogram of Returns', 'Returns', 'Density', os.path.join(PLOTS_DIR, 'A1.1.1.png'), mean=mean_returns)

    # A1.1.2 Asses Autocorrelation of Returns
    acf_values_returns, pacf_values_returns = summary_autocorrelation_assessment(df['R'])
    logging.info(f"A1.1.2 ACF values of Returns: {acf_values_returns}")
    logging.info(f"A1.1.2 PACF values of Returns: {pacf_values_returns}")
    plot_acf(df['R'], 'ACF of Returns', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.1.2_ACF.png'), lags=20)
    plot_pacf(df['R'], 'PACF of Returns', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.1.2_PACF.png'), lags=20)

    # A1.1.3 Asses Autocorrelation of Squared and Absolute Returns
    squared_returns = df['R']**2
    absolute_returns = abs(df['R'])
    acf_values_squared_returns, _ = summary_autocorrelation_assessment(squared_returns)
    acf_values_absolute_returns, _ = summary_autocorrelation_assessment(absolute_returns)
    
    logging.info(f"A1.1.3 ACF values of Squared Returns: {acf_values_squared_returns}")
    logging.info(f"A1.1.3 ACF values of Absolute Returns: {acf_values_absolute_returns}")
    plot_acf(squared_returns, 'ACF of Squared Returns', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.1.3_ACF_Squared_Returns.png'), lags=20)
    plot_acf(absolute_returns, 'ACF of Absolute Returns', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.1.3_ACF_Absolute_Returns.png'), lags=20)
    plot_pacf(squared_returns, 'PACF of Squared Returns', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.1.3_PACF_Squared_Returns.png'), lags=20)
    plot_pacf(absolute_returns, 'PACF of Absolute Returns', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.1.3_PACF_Absolute_Returns.png'), lags=20)

    # A1.1.4 Standardised Returns
    standardised_returns = df['R'] / np.sqrt(df['RV'])
    standardised_returns_mean, standardised_returns_std, standardised_returns_kurtosis, \
        standardised_returns_skew, standardised_returns_jarque_bera_test, _, _ = summary_normality_assessment(standardised_returns)
    log_normality_assessment(standardised_returns_mean, standardised_returns_std, standardised_returns_kurtosis, 
                             standardised_returns_skew, standardised_returns_jarque_bera_test, 
                             prefix="A1.1.4", variable_name="Standardised Returns")
    plot_histogram(standardised_returns, 'Histogram of Standardised Returns', 'Standardised Returns', 'Density', os.path.join(PLOTS_DIR, 'A1.1.4.png'), mean=standardised_returns_mean)



    #----------------------------------------#
    # A1.1 (Part II)Properties of Realised Variance 
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.1 (Part II)Properties of Realised Variance \n----------------------------------------")

    # A1.1.5 Asses Normality of Realised Variance
    mean_realised_variance, std_realised_variance, kurtosis_realised_variance, skew_realised_variance, \
        jarque_bera_test_realised_variance, acf_values_realised_variance, pacf_values_realised_variance = summary_normality_assessment(df['RV'])
    log_normality_assessment(mean_realised_variance, std_realised_variance, kurtosis_realised_variance, 
                             skew_realised_variance, jarque_bera_test_realised_variance, 
                             prefix="A1.1.5", variable_name="Realised Variance")

    # A1.1.6 Asses Autocorrelation of Realised Variance
    acf_values_realised_variance, pacf_values_realised_variance = summary_autocorrelation_assessment(df['RV'])
    logging.info(f"A1.1.6 ACF values of Realised Variance: {acf_values_realised_variance}")
    logging.info(f"A1.1.6 PACF values of Realised Variance: {pacf_values_realised_variance}")
    plot_acf(df['RV'], 'ACF of Realised Variance', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.1.6_ACF_Realised_Variance.png'), lags=20)
    plot_pacf(df['RV'], 'PACF of Realised Variance', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.1.6_PACF_Realised_Variance.png'), lags=20)

    # A1.1.7 Log Realised Variance
    log_realised_variance = np.log(df['RV'])
    log_realised_variance_mean, log_realised_variance_std, log_realised_variance_kurtosis, \
        log_realised_variance_skew, log_realised_variance_jarque_bera_test, log_realised_variance_acf_values, log_realised_variance_pacf_values = summary_normality_assessment(log_realised_variance)
    log_normality_assessment(log_realised_variance_mean, log_realised_variance_std, log_realised_variance_kurtosis, 
                             log_realised_variance_skew, log_realised_variance_jarque_bera_test, 
                             prefix="A1.1.7", variable_name="Log Realised Variance")
    plot_histogram(log_realised_variance, 'Histogram of Log Realised Variance', 'Log Realised Variance', 'Density', os.path.join(PLOTS_DIR, 'A1.1.7_Log_Realised_Variance.png'), mean=log_realised_variance_mean)

    # A1.1.8 Autocorrelation of Log Realised Variance
    logging.info(f"A1.1.8 ACF values of Log Realised Variance: {log_realised_variance_acf_values}")
    logging.info(f"A1.1.8 PACF values of Log Realised Variance: {log_realised_variance_pacf_values}")
    plot_acf(log_realised_variance, 'ACF of Log Realised Variance', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.1.8_ACF_Log_Realised_Variance.png'), lags=20)
    plot_pacf(log_realised_variance, 'PACF of Log Realised Variance', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.1.8_PACF_Log_Realised_Variance.png'), lags=20)


    #----------------------------------------#
    # A1.2 Historical Volatility
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.2 Historical Volatility \n----------------------------------------")

    T = [63, 125, 250]
    historical_variance = {63: [], 125: [], 250: []}
    for i in T:
        for j in range(i, len(df['R'])):
            returns_window = df['R'].iloc[j-i:j].values
            var_t = np.mean(returns_window**2)
            historical_variance[i].append(var_t)
    mean_historical_variance = {63: np.mean(historical_variance[63]), 125: np.mean(historical_variance[125]), 250: np.mean(historical_variance[250])}
    std_historical_variance = {63: np.std(historical_variance[63]), 125: np.std(historical_variance[125]), 250: np.std(historical_variance[250])}
    logging.info(f"A1.2 Mean Historical Variance for T=63: {mean_historical_variance[63]:.6f}")
    logging.info(f"A1.2 Mean Historical Variance for T=125: {mean_historical_variance[125]:.6f}")
    logging.info(f"A1.2 Mean Historical Variance for T=250: {mean_historical_variance[250]:.6f}")
    logging.info(f"A1.2 Std Historical Variance for T=63: {std_historical_variance[63]:.6f}")
    logging.info(f"A1.2 Std Historical Variance for T=125: {std_historical_variance[125]:.6f}")
    logging.info(f"A1.2 Std Historical Variance for T=250: {std_historical_variance[250]:.6f}")

    # A1.2.1 Plot Historical Variance
    # Align all series to start from index 250 (latest start) for comparison
    # T=63: skip first (250-63)=187 elements, T=125: skip first (250-125)=125 elements, T=250: use all
    plot_multiple_timeseries(
        [historical_variance[63][250-63:], historical_variance[125][250-125:], historical_variance[250]],
        ['Historical Variance for T=63', 'Historical Variance for T=125', 'Historical Variance for T=250'],
        'Historical Variance',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.2.1_Historical_Variance.png'),
        x_data=df['DATE'].iloc[250:].values
    )

    #----------------------------------------#
    # A1.3 Riskmetric Model
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.3 Riskmetric Model \n----------------------------------------")

    # A1.3.1 Initialise simga 0 & lambda
    sigma_0 = np.mean(df['R'].iloc[:63]**2)
    logging.info(f"A1.3.1 Initialise sigma 0: {sigma_0:.6f}")
    lambda_ = 0.94
    logging.info(f"A1.3.1 Initialise lambda: {lambda_:.6f}")

    # A1.3.2 Initialise Riskmetric Volatility
    riskmetric_volatility = [sigma_0]
    for t in range(63, len(df['R'])):  # t=63 corresponds to first day after sigma_0
        var_t = lambda_ * riskmetric_volatility[-1] + (1 - lambda_) * df['R'].iloc[t]**2
        riskmetric_volatility.append(var_t)
    
    # A1.3.3 Calculate Mean and Std of Riskmetric Volatility
    mean_riskmetric_volatility = np.mean(riskmetric_volatility)
    std_riskmetric_volatility = np.std(riskmetric_volatility)
    logging.info(f"A1.3.3 Mean Riskmetric Volatility: {mean_riskmetric_volatility:.6f}")
    logging.info(f"A1.3.3 Std Riskmetric Volatility: {std_riskmetric_volatility:.6f}")
    
    # A1.3.4 Plot Riskmetric Volatility
    plot_timeseries(riskmetric_volatility[1:], 'Riskmetric Volatility', 'Date', 'Volatility', 
                    os.path.join(PLOTS_DIR, 'A1.3.3_Riskmetric_Volatility.png'), 
                    label='Riskmetric Volatility', x_data=df['DATE'].iloc[63:].values)
    
    # A1.3.5 Plot Realised Variance vs Riskmetric Volatility (on same plot)
    plot_multiple_timeseries(
        [df['RV'].iloc[63:].values, riskmetric_volatility[1:]],
        ['Realised Variance', 'Riskmetric Volatility'],
        'Realised Variance vs Riskmetric Volatility',
        'Date',
        'Value',
        os.path.join(PLOTS_DIR, 'A1.3.5_Realised_Variance_vs_Riskmetric_Volatility.png'),
        x_data=df['DATE'].iloc[63:].values
    )



    #----------------------------------------#
    # A1.4 GARCH(1,1) Model
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.4 \n----------------------------------------")

    # A1.4.1 Sample Period 
    sample_period = df['R'].iloc[:1258]
    logging.info(f"A1.4.1 Sample Period: {sample_period.shape[0]} observations")

    # A1.4.1 Fit GARCH(1,1) Model
    model = arch_model(sample_period, vol='GARCH', p=1, q=1, dist='normal')
    result = model.fit(disp='off')
    logging.info(f"A.1.4.1 GARCH(1,1) Model Results: {result.summary()}")

    # A1.4.2 Plot GARCH(1,1) Model Results
    plot_timeseries(result.conditional_volatility, 'GARCH(1,1) Model Conditional Volatility', 'Date', 'Volatility', 
                    os.path.join(PLOTS_DIR, 'A1.4.2_GARCH_1_1_Model_Conditional_Volatility.png'), 
                    label='Conditional Volatility', x_data=df['DATE'].iloc[1258:].values)


    # A1.4.3 Plot Historical Variance vs Garch(1,1) Model Conditional Volatility vs Riskmetric Volatility
    print(len(historical_variance[63][1258-63:]))
    print(len(result.conditional_volatility))
    print(len(riskmetric_volatility[1258-62:]))
    print(len(df['DATE'].iloc[1258:]))
    plot_multiple_timeseries(
        [historical_variance[63][1258-65:], result.conditional_volatility, riskmetric_volatility[1258-66:]],
        ['Historical Variance', 'GARCH(1,1) Model Conditional Volatility', 'Riskmetric Volatility'],
        'Historical Variance vs GARCH(1,1) Model Conditional Volatility vs Riskmetric Volatility',
        'Date',
        'Value',
        os.path.join(PLOTS_DIR, 'A1.4.3_Historical_Variance_vs_GARCH_1_1_Model_Conditional_Volatility_vs_Riskmetric_Volatility.png'),
        x_data=df['DATE'].iloc[1258:].values
    )




if __name__ == "__main__":
    setup_logging()
    main()



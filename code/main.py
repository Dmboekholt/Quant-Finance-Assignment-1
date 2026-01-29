"""
Machine Learning Assignment
Main script for data processing and model training
"""
# Basic Imports
import os
from re import L
import sys
import arch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
from matplotlib import pyplot as plt
from arch import arch_model


# Import configuration
from config import DATA_DIR, PLOTS_DIR, setup_logging

# Import logging
import logging

# Import summary statistics
from Testing.summary import summary_normality_assessment, summary_autocorrelation_assessment, summary_basic_statistics
from Plotting.plots import plot_histogram, plot_acf, plot_pacf, plot_timeseries, plot_multiple_timeseries
from Logging.normality import log_normality_assessment
from evaluation import evaluate_models,calculate_MSPE, calculate_var_estimates, calculate_var_violations, christoffersen_uc_test, christoffersen_ind_test
from sklearn.metrics import mean_squared_error
from GARCHX import estimate_garchx, compute_garchx_variances, compute_garchx_standard_errors, create_garchx_result_object

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

    T = [63, 126, 252]  
    historical_variance = {63: [], 126: [], 252: []} 
    for i in T:
        for j in range(i, len(df['R'])):
            returns_window = df['R'].iloc[j-i:j].values
            var_t = np.mean(returns_window**2)
            historical_variance[i].append(var_t)
    mean_historical_variance = {63: np.mean(historical_variance[63]), 126: np.mean(historical_variance[126]), 252: np.mean(historical_variance[252])}
    std_historical_variance = {63: np.std(historical_variance[63]), 126: np.std(historical_variance[126]), 252: np.std(historical_variance[252])}
    logging.info(f"A1.2 Mean Historical Variance for T=63: {mean_historical_variance[63]:.6f}")
    logging.info(f"A1.2 Mean Historical Variance for T=125: {mean_historical_variance[126]:.6f}")
    logging.info(f"A1.2 Mean Historical Variance for T=250: {mean_historical_variance[252]:.6f}")
    logging.info(f"A1.2 Std Historical Variance for T=63: {std_historical_variance[63]:.6f}")
    logging.info(f"A1.2 Std Historical Variance for T=125: {std_historical_variance[126]:.6f}")
    logging.info(f"A1.2 Std Historical Variance for T=250: {std_historical_variance[252]:.6f}")

    # A1.2.1 Plot Historical Variance
    # Align all series to start from index 250 (latest start) for comparison
    # T=63: skip first (250-63)=187 elements, T=125: skip first (250-125)=125 elements, T=250: use all
    plot_multiple_timeseries(
        [historical_variance[63][252-63:], historical_variance[126][252-126:], historical_variance[252]],
        ['Historical Variance for T=63', 'Historical Variance for T=125', 'Historical Variance for T=250'],
        'Historical Variance',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.2.1_Historical_Variance.png'),
        x_data=df['DATE'].iloc[252:].values
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
    # A1.4 GARCH(1,1) Model, Assuming Normal Distribution of zt 
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.4 GARCH(1,1) Model, Assuming Normal Distribution of zt \n----------------------------------------")

    # A1.4.1 Sample Period & Fit GARCH(1,1) Model
    sample_period = df['R'].iloc[:1258]
    logging.info(f"A1.4.1 Sample Period: {sample_period.shape[0]} observations")
    model = arch_model(sample_period, mean = 'Constant', vol='GARCH', p=1, q=1, dist='normal')
    result_garch = model.fit(disp='off')
    logging.info(f"A.1.4.1 GARCH(1,1) Model Results: {result_garch.summary()}")

    # A1.4.2 Plot GARCH(1,1) Model Results (during estimation period)
    plot_timeseries(result_garch.conditional_volatility, 'GARCH(1,1) Model Conditional Volatility', 'Date', 'Volatility', 
                    os.path.join(PLOTS_DIR, 'A1.4.2_GARCH_1_1_Model_Conditional_Volatility.png'), 
                    label='Conditional Volatility', x_data=df['DATE'].iloc[:1258].values)

    # A1.4.3 Plot Historical Variance vs GARCH(1,1) Model Conditional Volatility vs Riskmetric Volatility
    # Align all series to estimation period (dates 63 to 1257) where all three have values
    # historical_variance[63] starts at date 63, so use indices 0 to (1258-63-1) = 0 to 1194
    # result.conditional_volatility has 1258 values for dates 0-1257, use indices 63 to 1257
    # riskmetric_volatility[1] is for date 63, so use indices 1 to (1258-63) = 1 to 1195
    # Note: Convert conditional_volatility (std dev) to variance (std dev^2) for proper comparison
    est_start_date = 63 
    est_end_date = 1258  
    plot_multiple_timeseries(
        [
            historical_variance[63][:est_end_date-est_start_date],  # From date 63 to 1257 (variance)
            result_garch.conditional_volatility[est_start_date:]**2,  # From date 63 to 1257 (converted to variance)
            riskmetric_volatility[1:est_end_date-est_start_date+1]  # From date 63 to 1257 (variance)
        ],
        ['Historical Variance (T=63)', 'GARCH(1,1) Model Conditional Variance', 'Riskmetric Variance'],
        'Historical Variance vs GARCH(1,1) Model Conditional Variance vs Riskmetric Variance',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.4.3_Historical_Variance_vs_GARCH_1_1_Model_Conditional_Volatility_vs_Riskmetric_Volatility.png'),
        x_data=df['DATE'].iloc[est_start_date:est_end_date].values
    )

    # A1.4.4 Standardised Returns
    mean = result_garch.params['mu']
    st_dev = result_garch.conditional_volatility
    zt = ( df['R'].iloc[:1258] - mean ) / st_dev
    
    # A1.4.5 Asses Normality of Standardised Returns
    mean_zt, std_zt, kurtosis_zt, skew_zt, jarque_bera_test_zt, acf_values_zt, pacf_values_zt = summary_normality_assessment(zt)
    log_normality_assessment(mean_zt, std_zt, kurtosis_zt, skew_zt, jarque_bera_test_zt, prefix="A1.4.4", variable_name="Standardised Returns")
    plot_histogram(zt, 'Histogram of Standardised Returns', 'Standardised Returns', 'Density', os.path.join(PLOTS_DIR, 'A1.4.4_Histogram_Standardised_Returns.png'), mean=mean_zt)
    plot_acf(zt, 'ACF of Standardised Returns', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.4.4_ACF_Standardised_Returns.png'), lags=20)
    plot_pacf(zt, 'PACF of Standardised Returns', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.4.4_PACF_Standardised_Returns.png'), lags=20)

    # A1.4.6 Asses Autocorrelation of Squared Standardised Returns
    zt_squared = zt**2
    plot_acf(zt_squared, 'ACF of Squared Standardised Returns', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.4.6_ACF_Squared_Standardised_Returns.png'), lags=20)
    plot_pacf(zt_squared, 'PACF of Squared Standardised Returns', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.4.6_PACF_Squared_Standardised_Returns.png'), lags=20)


    #----------------------------------------#
    # A1.5 ARCH(1) Model, Assuming Normal Distribution of zt 
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.5 ARCH(1) Model, Assuming Normal Distribution of zt \n----------------------------------------")

    # A1.5.1 Sample Period & Fit ARCH(1) Model
    sample_period = df['R'].iloc[:1258]
    logging.info(f"A1.5.1 Sample Period: {sample_period.shape[0]} observations")
    model = arch_model(sample_period, mean='Constant', vol='ARCH', q=1, dist='normal')
    result_arch = model.fit(disp='off')
    logging.info(f"A.1.5.1 ARCH(1) Model Results: {result_arch.summary()}")

    # A1.5.2 Plot ARCH(1) Model Results (during estimation period)
    plot_timeseries(result_arch.conditional_volatility, 'ARCH(1) Model Conditional Volatility', 'Date', 'Volatility', 
                    os.path.join(PLOTS_DIR, 'A1.5.2_ARCH_1_Model_Conditional_Volatility.png'), 
                    label='Conditional Volatility', x_data=df['DATE'].iloc[:1258].values)
    
    # A1.5.3 Plot Historical Variance vs ARCH(1) Model Conditional Volatility vs Riskmetric Volatility
    # Align all series to estimation period (dates 63 to 1257) where all three have values
    # historical_variance[63] starts at date 63, so use indices 0 to (1258-63-1) = 0 to 1194
    # result.conditional_volatility has 1258 values for dates 0-1257, use indices 63 to 1257
    # riskmetric_volatility[1] is for date 63, so use indices 1 to (1258-63) = 1 to 1195
    # Note: Convert conditional_volatility (std dev) to variance (std dev^2) for proper comparison
    est_start_date = 63 
    est_end_date = 1258  
    plot_multiple_timeseries(
        [
            historical_variance[63][:est_end_date-est_start_date],  # From date 63 to 1257 (variance)
            result_arch.conditional_volatility[est_start_date:]**2,  # From date 63 to 1257 (converted to variance)
            riskmetric_volatility[1:est_end_date-est_start_date+1]  # From date 63 to 1257 (variance)
        ],
        ['Historical Variance (T=63)', 'ARCH(1) Model Conditional Variance', 'Riskmetric Variance'],
        'Historical Variance vs ARCH(1) Model Conditional Variance vs Riskmetric Variance',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.5.3_Historical_Variance_vs_ARCH_1_Model_Conditional_Volatility_vs_Riskmetric_Volatility.png'),
        x_data=df['DATE'].iloc[est_start_date:est_end_date].values
    )

    # A1.5.4 Standardised Returns
    mean = result_arch.params['mu']
    st_dev = result_arch.conditional_volatility
    zt = ( df['R'].iloc[:1258] - mean ) / st_dev
    
    # A1.5.5 Asses Normality of Standardised Returns
    mean_zt, std_zt, kurtosis_zt, skew_zt, jarque_bera_test_zt, acf_values_zt, pacf_values_zt = summary_normality_assessment(zt)
    log_normality_assessment(mean_zt, std_zt, kurtosis_zt, skew_zt, jarque_bera_test_zt, prefix="A1.5.5", variable_name="Standardised Returns")
    plot_histogram(zt, 'Histogram of Standardised Returns', 'Standardised Returns', 'Density', os.path.join(PLOTS_DIR, 'A1.5.4_Histogram_Standardised_Returns.png'), mean=mean_zt)
    plot_acf(zt, 'ACF of Standardised Returns', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.5.5_ACF_Standardised_Returns.png'), lags=20)
    plot_pacf(zt, 'PACF of Standardised Returns', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.5.4_PACF_Standardised_Returns.png'), lags=20)

    # A1.5.6 Asses Autocorrelation of Squared Standardised Returns
    zt_squared = zt**2
    plot_acf(zt_squared, 'ACF of Squared Standardised Returns', 'Lags', 'ACF', os.path.join(PLOTS_DIR, 'A1.5.6_ACF_Squared_Standardised_Returns.png'), lags=20)
    plot_pacf(zt_squared, 'PACF of Squared Standardised Returns', 'Lags', 'PACF', os.path.join(PLOTS_DIR, 'A1.5.6_PACF_Squared_Standardised_Returns.png'), lags=20)

    # Plot Arch(1) vs Garch(1,1)
    plot_multiple_timeseries(
        [
            result_garch.conditional_volatility[est_start_date:]**2,  # From date 63 to 1257 (converted to variance)
            result_arch.conditional_volatility[est_start_date:]**2,  # From date 63 to 1257 (converted to variance)
        ],
        ['GARCH(1,1) Model Conditional Variance', 'ARCH(1) Model Conditional Variance'],
        'GARCH(1,1) Model Conditional Variance vs ARCH(1) Model Conditional Variance',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.5.7_GARCH_1_1_Model_Conditional_Variance_vs_ARCH_1_Model_Conditional_Variance.png'),
        x_data=df['DATE'].iloc[est_start_date:est_end_date].values
    )

    #----------------------------------------#
    # A1.6 Threshold GARCH (asymmetric) and GARCH-X
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.6 Threshold GARCH (asymmetric) and GARCH-X \n----------------------------------------")
    sample_period = df['R'].iloc[:1258]
    logging.info(f"A1.6.1 Sample Period: {sample_period.shape[0]} observations")


   # A1.6.1 Threshold GARCH (asymmetric)
    threshold_garch_model = arch_model(
        sample_period,
        mean='Constant',
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal"
    )
    threshold_garch_result = threshold_garch_model.fit(disp='off')
    logging.info(f"A1.6.2 Threshold Garch Model Results:\n{threshold_garch_result.summary()}")

    # A1.6.2 GARCH-X
    logging.info("Estimating GARCH-X model...")

    rv_sample = df['RV'].iloc[:1258]
    exog = rv_sample.shift(1)
    data = pd.DataFrame({'R': sample_period, 'RV_lag1': exog}).dropna()

    y = data['R'].values
    X = data['RV_lag1'].values

    # Estimate
    garchx_opt_result = estimate_garchx(y, X)

    if not garchx_opt_result.success:
        logging.warning(f"GARCH-X optimization warning: {garchx_opt_result.message}")

    # Extract parameters
    params = garchx_opt_result.x
    mu, omega, alpha, beta, delta = params

    # Compute conditional variances
    sigma2 = compute_garchx_variances(params, y, X)

    # Compute standard errors
    std_errors = compute_garchx_standard_errors(params, y, X)

    # Log likelihood
    loglik = -garchx_opt_result.fun

    # Create result object
    garchx_result = create_garchx_result_object(params, std_errors, sigma2, loglik)

    # Log results
    logging.info(f"A1.6.3 GARCHX Model Results:")
    logging.info(f"  Parameters:")
    logging.info(f"    mu:    {mu:.6f} (std err: {std_errors[0]:.6f})")
    logging.info(f"    omega: {omega:.6f} (std err: {std_errors[1]:.6f})")
    logging.info(f"    alpha: {alpha:.6f} (std err: {std_errors[2]:.6f})")
    logging.info(f"    beta:  {beta:.6f} (std err: {std_errors[3]:.6f})")
    logging.info(f"    delta: {delta:.6f} (std err: {std_errors[4]:.6f})")
    logging.info(f"  alpha + beta: {alpha + beta:.6f}")
    logging.info(f"  Log-likelihood: {loglik:.2f}")
    # Check if delta is significant
    if not np.isnan(std_errors[4]):
        t_stat = delta / std_errors[4]
        logging.info(f"  Delta t-statistic: {t_stat:.3f} (significant if |t| > 1.96)")

    # A1.6.3 - Inspect models for  Dec 2018 return
    logging.info(f"A1.6.3 Log 26 Jan 2016 return: {df[df['DATE'] == '2016-01-26']['R'].values[0]:.6f}")
    Garch_conditional_volatility_26_Jan_2016 = threshold_garch_result.conditional_volatility[df[df['DATE'] == '2016-01-26'].index[0]]
    Threshold_Garch_conditional_volatility_26_Jan_2016 = threshold_garch_result.conditional_volatility[df[df['DATE'] == '2016-01-26'].index[0]]
    logging.info(f"A1.6.3 Threshold Garch Conditional Volatility 26 Jan 2016: {Threshold_Garch_conditional_volatility_26_Jan_2016:.6f}")
    logging.info(f"A1.6.3 Garch Conditional Volatility 26 Jan 2016: {Garch_conditional_volatility_26_Jan_2016:.6f}")

    # A1.6.3 - Inspect models for 14 Dec 2018 return
    logging.info(f"A1.6.3 Log 14 Dec 2018 return: {df[df['DATE'] == '2018-12-14']['R'].values[0]:.6f}")
    Garch_conditional_volatility_14_Dec_2018 = threshold_garch_result.conditional_volatility[df[df['DATE'] == '2018-12-14'].index[0]]
    Threshold_Garch_conditional_volatility_14_Dec_2018 = threshold_garch_result.conditional_volatility[df[df['DATE'] == '2018-12-14'].index[0]]
    logging.info(f"A1.6.3 Threshold Garch Conditional Volatility 14 Dec 2018: {Threshold_Garch_conditional_volatility_14_Dec_2018:.6f}")
    logging.info(f"A1.6.3 Garch Conditional Volatility 14 Dec 2018: {Garch_conditional_volatility_14_Dec_2018:.6f}")


    #----------------------------------------#
    # A1.7 Forecasting Volatility
    #----------------------------------------#
    logging.info("\n----------------------------------------\n A1.7 Forecasting Volatility \n----------------------------------------")

    forecast_period = df['R'].iloc[1258:]
    logging.info(f"A1.7.1 Forecast Period: {forecast_period.shape[0]} observations")
    variance_proxy = df['R'].iloc[1258:]**2
    logging.info(f"A1.7.1 Variance Proxy (Squared Returns): {variance_proxy.shape[0]} observations")
    historical_var_forecast = historical_variance[126][1258-126:]  
    riskmetric_var_forecast = riskmetric_volatility[1+1258-63:]
    logging.info("Generating rolling one-step-ahead forecasts for GARCH models...")

    # Use evaluate_models function to generate one-step-ahead forecasts
    forecasts = evaluate_models(variance_proxy, df, horizon=1)
    garch_var_forecast = forecasts['GARCH']
    arch_var_forecast = forecasts['ARCH']
    threshold_garch_var_forecast = forecasts['Threshold_GARCH']
    garchx_var_forecast = forecasts['GARCH_X']

    # Calculate MSPE for each model using squared returns proxy
    logging.info("\n=== A1.7 Part 1: MSPE using Squared Returns as proxy ===")

    # Create forecasts dictionary for all models
    all_forecasts = {
        'Historical_Variance_T126': historical_var_forecast,
        'Riskmetric': riskmetric_var_forecast,
        'ARCH': arch_var_forecast,
        'GARCH': garch_var_forecast,
        'Threshold_GARCH': threshold_garch_var_forecast,
        'GARCH_X': garchx_var_forecast
    }

    mspe_results_squared = {}
    for model_name, predicted in all_forecasts.items():
        mspe = calculate_MSPE(variance_proxy, predicted, model_name=model_name)
        mspe_results_squared[model_name] = mspe

    # Rank models by MSPE
    logging.info("\nRanking of models by MSPE (Squared Returns proxy):")
    ranked_models = sorted(mspe_results_squared.items(), key=lambda x: x[1])
    for rank, (model, mspe) in enumerate(ranked_models, 1):
        logging.info(f"  {rank}. {model}: {mspe:.6f}")

    # Get realised variance for comparison
    variance_proxy_realised = df['RV'].iloc[1258:]

    # A1.7.2 Calculate MSPE using Realised Variance as proxy
    logging.info("\n=== A1.7 Part 2: MSPE using Realised Variance (RV) as proxy ===")

    mspe_results_rv = {}
    for model_name, predicted in all_forecasts.items():
        mspe = calculate_MSPE(variance_proxy_realised, predicted, model_name=f"{model_name} (RV)")
        mspe_results_rv[model_name] = mspe

    # Rank models by MSPE with RV proxy
    logging.info("\nRanking of models by MSPE (Realized Variance proxy):")
    ranked_models_rv = sorted(mspe_results_rv.items(), key=lambda x: x[1])
    for rank, (model, mspe) in enumerate(ranked_models_rv, 1):
        logging.info(f"  {rank}. {model}: {mspe:.6f}")

    # Compare rankings
    logging.info("\n=== Comparison of Rankings ===")
    logging.info("Differences in model rankings between proxies:")
    for i, ((model1, _), (model2, _)) in enumerate(zip(ranked_models, ranked_models_rv), 1):
        if model1 != model2:
            logging.info(f"  Rank {i}: {model1} (Squared Returns) vs {model2} (RV)")

    # Plot all models together
    plot_multiple_timeseries(
        [
            variance_proxy_realised.values,
            historical_var_forecast,
            riskmetric_var_forecast,
            garch_var_forecast,
            arch_var_forecast,
            threshold_garch_var_forecast,
            garchx_var_forecast
        ],
        [
            'Realised Variance',
            'Historical Variance T=126',
            'Riskmetric',
            'GARCH(1,1)',
            'ARCH(1)',
            'Threshold GARCH',
            'GARCH-X'
        ],
        'Variance Forecasts vs Realised Variance (Full Forecast Period)',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.7_All_Forecasts_Full_Period.png'),
        x_data=df['DATE'].iloc[1258:].values
    )

    #----------------------------------------#
    # A1.7 Part 3: January 9 - February 17, 2023 Analysis
    #----------------------------------------#
    logging.info("\n=== A1.7 Part 3: Analysis of January 9 - February 17, 2023 ===")

    # Find indices for this period
    jan_start = df[df['DATE'] == '2023-01-09'].index[0]
    feb_end = df[df['DATE'] == '2023-02-17'].index[0]

    # Find January 24, 2023
    jan_24_idx = df[df['DATE'] == '2023-01-24'].index[0]

    logging.info(f"Period: {df['DATE'].iloc[jan_start]} to {df['DATE'].iloc[feb_end]}")
    logging.info(f"Important date: January 24, 2023 (index {jan_24_idx})")
    logging.info(f"Return on Jan 24, 2023: {df['R'].iloc[jan_24_idx]:.4f}%")
    logging.info(f"Realized Variance on Jan 24, 2023: {df['RV'].iloc[jan_24_idx]:.4f}")

    # Extract data for this period (relative to forecast period start at 1258)
    period1_start_rel = jan_start - 1258
    period1_end_rel = feb_end - 1258 + 1

    period1_dates = df['DATE'].iloc[jan_start:feb_end+1].values
    period1_rv = variance_proxy_realised.iloc[period1_start_rel:period1_end_rel].values
    period1_garch = garch_var_forecast[period1_start_rel:period1_end_rel]
    period1_tgarch = threshold_garch_var_forecast[period1_start_rel:period1_end_rel]
    period1_garchx = garchx_var_forecast[period1_start_rel:period1_end_rel]

    # Plot for January-February 2023 period
    plot_multiple_timeseries(
        [period1_rv, period1_garch, period1_tgarch, period1_garchx],
        ['Realised Variance', 'GARCH(1,1)', 'Threshold GARCH', 'GARCH-X'],
        'Variance Forecasts: January 9 - February 17, 2023\n(Around January 24, 2023 Event)',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.7.3_Jan_Feb_2023_Period.png'),
        x_data=period1_dates
    )

    logging.info(f"\nAnalysis around January 24, 2023:")
    jan24_rel = jan_24_idx - 1258
    for i in range(max(0, jan24_rel-2), min(len(garch_var_forecast), jan24_rel+5)):
        date_idx = 1258 + i
        logging.info(f"  {df['DATE'].iloc[date_idx]}: RV={variance_proxy_realised.iloc[i]:.4f}, "
                    f"GARCH={garch_var_forecast[i]:.4f}, TGARCH={threshold_garch_var_forecast[i]:.4f}, "
                    f"GARCHX={garchx_var_forecast[i]:.4f}")

    #----------------------------------------#
    # A1.7 Part 4: July 3 - August 11, 2023 Analysis
    #----------------------------------------#
    logging.info("\n=== A1.7 Part 4: Analysis of July 3 - August 11, 2023 ===")

    # Find indices for this period
    jul_start = df[df['DATE'] == '2023-07-03'].index[0]
    aug_end = df[df['DATE'] == '2023-08-11'].index[0]

    logging.info(f"Period: {df['DATE'].loc[jul_start]} to {df['DATE'].loc[aug_end]}")

    # Extract data for this period
    period2_start_rel = jul_start - 1258
    period2_end_rel = aug_end - 1258 + 1

    # Find day with highest return in this period
    period2_returns = df['R'].loc[jul_start:aug_end]
    max_return_idx = period2_returns.idxmax()  # DataFrame index
    max_return_date = df.loc[max_return_idx, 'DATE']
    max_return_value = df.loc[max_return_idx, 'R']

    logging.info(f"Highest return in period: {max_return_value:.4f}% on {max_return_date}")
    logging.info(f"Realized Variance on that day: {df.loc[max_return_idx, 'RV']:.4f}")

    period2_dates = df['DATE'].loc[jul_start:aug_end].values
    period2_rv = variance_proxy_realised.iloc[period2_start_rel:period2_end_rel].values
    period2_garch = garch_var_forecast[period2_start_rel:period2_end_rel]
    period2_tgarch = threshold_garch_var_forecast[period2_start_rel:period2_end_rel]
    period2_garchx = garchx_var_forecast[period2_start_rel:period2_end_rel]

    # Plot for July-August 2023 period
    plot_multiple_timeseries(
        [period2_rv, period2_garch, period2_tgarch, period2_garchx],
        ['Realised Variance', 'GARCH(1,1)', 'Threshold GARCH', 'GARCH-X'],
        f'Variance Forecasts: July 3 - August 11, 2023\n(Highest return: {max_return_value:.2f}% on {max_return_date})',
        'Date',
        'Variance',
        os.path.join(PLOTS_DIR, 'A1.7.4_Jul_Aug_2023_Period.png'),
        x_data=period2_dates
    )

    logging.info(f"\nAnalysis around highest return day ({max_return_date}):")
    max_return_rel = int(max_return_idx) - 1258

    for i in range(max(0, max_return_rel-2), min(len(garch_var_forecast), max_return_rel+5)):
        date_idx = 1258 + i
        logging.info(f"  {df['DATE'].loc[date_idx]}: RV={variance_proxy_realised.iloc[i]:.4f}, "
                    f"GARCH={garch_var_forecast[i]:.4f}, TGARCH={threshold_garch_var_forecast[i]:.4f}, "
                    f"GARCHX={garchx_var_forecast[i]:.4f}")

    #----------------------------------------#
    # A1.8 Value-at-Risk (VaR) Analysis
    #----------------------------------------#
    logging.info("\n========================================")
    logging.info(" A1.8 Value-at-Risk (VaR) Analysis")
    logging.info("========================================")

    logging.info("\nConstructing 1-day ahead VaR estimates at 90%, 95%, and 99% confidence levels")
    logging.info("for the period January 2, 2019 - December 31, 2023")

    # Prepare variance forecasts for VaR calculation
    variance_forecasts_all = {
        'Historical_Variance_T126': historical_var_forecast,
        'Riskmetric': riskmetric_var_forecast,
        'ARCH': arch_var_forecast,
        'GARCH': garch_var_forecast,
        'Threshold_GARCH': threshold_garch_var_forecast,
        'GARCH_X': garchx_var_forecast
    }

    # Calculate VaR estimates
    confidence_levels = [0.90, 0.95, 0.99]
    var_estimates = calculate_var_estimates(forecast_period.values, variance_forecasts_all, confidence_levels)
    logging.info("✓ VaR estimates calculated for all models and confidence levels")

    # Calculate VaR violations (backtesting)
    var_violations = calculate_var_violations(forecast_period.values, var_estimates, confidence_levels)

    # Log VaR violation statistics
    logging.info("\n" + "="*60)
    logging.info("VaR VIOLATION STATISTICS")
    logging.info("="*60)

    for confidence in confidence_levels:
        logging.info(f"\n{'='*60}")
        logging.info(f"  Confidence Level: {confidence*100:.0f}%")
        logging.info(f"  Expected Violation Rate: {(1-confidence)*100:.2f}%")
        logging.info(f"{'='*60}")
        
        for model_name, stats_dict in var_violations[confidence].items():
            actual_rate = stats_dict['violation_rate'] * 100
            expected_rate = (1 - confidence) * 100
            logging.info(f"\n  {model_name}:")
            logging.info(f"    Violations: {stats_dict['breaches']:.0f} / {len(forecast_period)}")
            logging.info(f"    Actual Rate: {actual_rate:.2f}%")
            logging.info(f"    Expected Rate: {expected_rate:.2f}%")
            logging.info(f"    Difference: {actual_rate - expected_rate:+.2f}%")

    # Christoffersen Unconditional Coverage (UC) Test
    logging.info("\n" + "="*80)
    logging.info("CHRISTOFFERSEN UNCONDITIONAL COVERAGE (UC) TEST")
    logging.info("="*80)
    logging.info("H0: The VaR model has correct unconditional coverage")
    logging.info("    (observed violation rate = expected violation rate)")
    logging.info("Reject H0 if p-value < 0.05 (model has incorrect coverage)\n")

    uc_summary = {}
    for confidence in confidence_levels:
        logging.info(f"\n{'─'*80}")
        logging.info(f"Confidence Level: {confidence*100:.0f}%")
        logging.info(f"{'─'*80}")
        
        uc_results = christoffersen_uc_test(forecast_period.values, var_estimates[confidence], confidence)
        uc_summary[confidence] = uc_results
        
        for model_name, test_results in uc_results.items():
            logging.info(f"\n  {model_name}:")
            logging.info(f"    Violations: {test_results['violations']:.0f} (Expected: {test_results['expected_violations']:.2f})")
            logging.info(f"    Violation Rate: {test_results['violation_rate']*100:.2f}% (Expected: {test_results['expected_rate']*100:.2f}%)")
            logging.info(f"    LR_UC Statistic: {test_results['LR_UC']:.4f}")
            logging.info(f"    p-value: {test_results['p_value']:.4f}")
            logging.info(f"    Reject H0 (α=0.05): {'YES - INCORRECT COVERAGE' if test_results['reject_h0'] else 'NO - Correct coverage'}")

    # Christoffersen Independence (Ind) Test
    logging.info("\n" + "="*80)
    logging.info("CHRISTOFFERSEN INDEPENDENCE TEST")
    logging.info("="*80)
    logging.info("H0: VaR violations are independently distributed (no clustering)")
    logging.info("Reject H0 if p-value < 0.05 (violations are clustered)\n")

    ind_summary = {}
    for confidence in confidence_levels:
        logging.info(f"\n{'─'*80}")
        logging.info(f"Confidence Level: {confidence*100:.0f}%")
        logging.info(f"{'─'*80}")
        
        ind_results = christoffersen_ind_test(forecast_period.values, var_estimates[confidence], confidence)
        ind_summary[confidence] = ind_results
        
        for model_name, test_results in ind_results.items():
            if not np.isnan(test_results['LR_Ind']):
                logging.info(f"\n  {model_name}:")
                logging.info(f"    Transition Matrix:")
                logging.info(f"      No viol → No viol: {test_results['transitions_00']}")
                logging.info(f"      No viol → Violation: {test_results['transitions_01']}")
                logging.info(f"      Violation → No viol: {test_results['transitions_10']}")
                logging.info(f"      Violation → Violation: {test_results['transitions_11']}")
                logging.info(f"    Total Violations: {test_results['total_violations']}")
                logging.info(f"    LR_Ind Statistic: {test_results['LR_Ind']:.4f}")
                logging.info(f"    p-value: {test_results['p_value']:.4f}")
                logging.info(f"    Reject H0 (α=0.05): {'YES - VIOLATIONS CLUSTERED' if test_results['reject_h0'] else 'NO - Independent violations'}")
            else:
                logging.info(f"\n  {model_name}: Unable to compute (insufficient violations)")

    # Summary table
    logging.info("\n" + "="*80)
    logging.info("SUMMARY OF CHRISTOFFERSEN TESTS")
    logging.info("="*80)

    for confidence in confidence_levels:
        logging.info(f"\n{confidence*100:.0f}% Confidence Level:")
        logging.info(f"{'Model':<25} {'UC Test':<15} {'Ind Test':<15} {'Overall'}")
        logging.info(f"{'-'*70}")
        
        for model_name in variance_forecasts_all.keys():
            uc_pass = "PASS" if not uc_summary[confidence][model_name]['reject_h0'] else "FAIL"
            
            if not np.isnan(ind_summary[confidence][model_name]['LR_Ind']):
                ind_pass = "PASS" if not ind_summary[confidence][model_name]['reject_h0'] else "FAIL"
            else:
                ind_pass = "N/A"
            
            overall = "PASS" if (uc_pass == "PASS" and (ind_pass == "PASS" or ind_pass == "N/A")) else "FAIL"
            
            logging.info(f"{model_name:<25} {uc_pass:<15} {ind_pass:<15} {overall}")

    logging.info("\n" + "="*80)
    logging.info("Note: A model PASSES if it fails to reject H0 (p-value >= 0.05)")
    logging.info("      UC Test: Correct unconditional coverage")
    logging.info("      Ind Test: Independent violations (no clustering)")
    logging.info("="*80)

    print(f"\nUsing arch version: {arch.__version__}")

"""
Complete GARCH-X implementation with debugging
Test this file separately first to make sure it works
"""

if __name__ == "__main__":  
    # Setup logging
    setup_logging()
    # Run main function
    main()



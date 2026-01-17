"""
Logging functions for normality assessment
Standard logging for statistical tests and normality checks
"""

import logging


def log_normality_assessment(mean, std, kurtosis_value, skew_value, jarque_bera_test, prefix="", variable_name=""):
    """
    Log normality assessment statistics in a standard format.
    
    Parameters:
    - mean: float, mean value
    - std: float, standard deviation
    - kurtosis_value: float, kurtosis value
    - skew_value: float, skewness value
    - jarque_bera_test: tuple, (test_statistic, p_value) from jarque_bera test
    - prefix: str, optional prefix for log messages (e.g., "A1.1.1")
    - variable_name: str, optional name of the variable being assessed (e.g., "Returns")
    
    Example:
        log_normality_assessment(mean, std, kurtosis, skew, jb_test, prefix="A1.1.1", variable_name="Returns")
    """
    # Build prefix string
    log_prefix = f"{prefix} " if prefix else ""
    var_name = f"of {variable_name}" if variable_name else ""
    
    # Log all normality statistics
    logging.info(f"{log_prefix}Mean {var_name}: {mean:.6f}")
    logging.info(f"{log_prefix}Standard Deviation {var_name}: {std:.6f}")
    logging.info(f"{log_prefix}Kurtosis {var_name}: {kurtosis_value:.6f}")
    logging.info(f"{log_prefix}Skew {var_name}: {skew_value:.6f}")
    logging.info(f"{log_prefix}Jarque-Bera test statistic {var_name}: {jarque_bera_test[0]:.6f}")
    logging.info(f"{log_prefix}Jarque-Bera test p-value {var_name}: {jarque_bera_test[1]:.6f}")


def log_normality_from_summary_stats(mean, std, kurtosis_value, skew_value, jarque_bera_test, prefix="", variable_name=""):
    """
    Convenience function that accepts statistics in the same order as summary_statistics returns.
    This is an alias for log_normality_assessment for consistency.
    """
    log_normality_assessment(mean, std, kurtosis_value, skew_value, jarque_bera_test, prefix, variable_name)


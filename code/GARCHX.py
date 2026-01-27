"""
GARCH-X Model Implementation
Complete implementation for GARCH-X estimation with proper lag handling

The GARCH-X(1,1) model specification:
σ²_t = ω + α(r_{t-1} - μ)² + βσ²_{t-1} + δRV_{t-1}

Key implementation details:
- After shift(1) and dropna, rv_lagged is already RV_{t-1} for each observation
- We use rv_lagged[t-1] in the loop to ensure proper temporal alignment
- This ensures RV_{t-1} is used to predict σ²_t
"""

import numpy as np
from scipy.optimize import minimize

def garchx_loglikelihood(params, returns, rv_lagged):
    """
    Negative log-likelihood for GARCH-X(1,1) model
    σ²_t = ω + α(r_{t-1} - μ)² + βσ²_{t-1} + δRV_{t-1}
    
    Parameters:
    -----------
    params : array [mu, omega, alpha, beta, delta]
    returns : array of returns (r_t)
    rv_lagged : array of lagged realized variance (RV_{t-1} for each t)
    
    Note: rv_lagged[t-1] gives RV_{t-1} which predicts σ²_t
    """
    mu, omega, alpha, beta, delta = params
    
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        # Variance equation: σ²_t = ω + α(r_{t-1}-μ)² + βσ²_{t-1} + δRV_{t-1}
        # rv_lagged[t-1] is RV_{t-1} (properly lagged)
        sigma2[t] = omega + alpha * (returns[t-1] - mu)**2 + beta * sigma2[t-1] + delta * rv_lagged[t-1]
        if sigma2[t] <= 0:
            sigma2[t] = 1e-8
    
    try:
        loglik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (returns - mu)**2 / sigma2)
        if np.isnan(loglik) or np.isinf(loglik):
            return 1e10
        return -loglik
    except:
        return 1e10


def estimate_garchx(returns, rv_lagged):
    """
    Estimate GARCH-X(1,1) model parameters using Maximum Likelihood
    
    Parameters:
    -----------
    returns : array of returns
    rv_lagged : array of lagged realized variance
    
    Returns:
    --------
    result : optimization result with estimated parameters
    """
    mu_init = np.mean(returns)
    var_unc = np.var(returns)
    
    # Starting values: [mu, omega, alpha, beta, delta]
    params_init = [mu_init, 0.01*var_unc, 0.05, 0.85, 0.05]
    
    bounds = [
        (None, None),           # mu (unrestricted)
        (1e-6, 10*var_unc),     # omega > 0
        (1e-6, 0.3),            # alpha > 0, < 1
        (0.5, 0.98),            # beta (high persistence)
        (-1, 1)                 # delta (can be negative or positive)
    ]
    
    # Constraint: alpha + beta < 1 for stationarity
    def stationarity_constraint(params):
        return 0.9999 - (params[2] + params[3])
    
    constraints = {'type': 'ineq', 'fun': stationarity_constraint}
    
    result = minimize(
        garchx_loglikelihood,
        params_init,
        args=(returns, rv_lagged),
        method='SLSQP',  # SLSQP handles constraints better
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-9}
    )
    
    return result


def compute_garchx_variances(params, returns, rv_lagged):
    """
    Compute conditional variances from estimated GARCH-X model
    
    Parameters:
    -----------
    params : array [mu, omega, alpha, beta, delta]
    returns : array of returns
    rv_lagged : array of lagged realized variance
    
    Returns:
    --------
    sigma2 : array of conditional variances
    """
    mu, omega, alpha, beta, delta = params
    
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        # Variance equation with proper lag
        sigma2[t] = omega + alpha * (returns[t-1] - mu)**2 + beta * sigma2[t-1] + delta * rv_lagged[t-1]
        sigma2[t] = max(sigma2[t], 1e-8)
    
    return sigma2


def compute_garchx_standard_errors(params, returns, rv_lagged):
    """
    Compute standard errors using numerical Hessian diagonal
    
    Parameters:
    -----------
    params : array [mu, omega, alpha, beta, delta]
    returns : array of returns
    rv_lagged : array of lagged realized variance
    
    Returns:
    --------
    std_errors : array of standard errors for each parameter
    """
    epsilon = 1e-5
    n = len(params)
    
    def f(p):
        return garchx_loglikelihood(p, returns, rv_lagged)
    
    # Compute Hessian diagonal (second derivatives)
    hess_diag = np.zeros(n)
    for i in range(n):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += epsilon
        p_minus[i] -= epsilon
        
        hess_diag[i] = (f(p_plus) - 2*f(params) + f(p_minus)) / (epsilon**2)
    
    # Standard errors from diagonal of inverse Hessian
    std_errors = np.zeros(n)
    for i in range(n):
        if hess_diag[i] > 0:
            std_errors[i] = 1.0 / np.sqrt(hess_diag[i])
        else:
            std_errors[i] = np.nan
    
    return std_errors


def create_garchx_result_object(params, std_errors, conditional_variances, loglik):
    """
    Create result object compatible with arch models
    
    Parameters:
    -----------
    params : array [mu, omega, alpha, beta, delta]
    std_errors : array of standard errors
    conditional_variances : array of conditional variances
    loglik : log-likelihood value
    
    Returns:
    --------
    result : object with params, std_errors, conditional_volatility attributes
    """
    mu, omega, alpha, beta, delta = params
    
    result = type('GARCHXResult', (object,), {
        'params': {
            'mu': mu, 
            'omega': omega, 
            'alpha[1]': alpha, 
            'beta[1]': beta, 
            'delta': delta
        },
        'std_errors': {
            'mu': std_errors[0],
            'omega': std_errors[1],
            'alpha[1]': std_errors[2],
            'beta[1]': std_errors[3],
            'delta': std_errors[4]
        },
        'conditional_volatility': np.sqrt(conditional_variances),
        'loglikelihood': loglik
    })()
    
    return result


def interpret_garchx_results(params, std_errors):
    """
    Helper function to interpret GARCH-X parameter estimates
    
    Parameters:
    -----------
    params : array [mu, omega, alpha, beta, delta]
    std_errors : array of standard errors
    
    Returns:
    --------
    interpretation : dict with parameter interpretations
    """
    mu, omega, alpha, beta, delta = params
    
    # Compute t-statistics
    t_stats = params / std_errors
    
    interpretation = {
        'persistence': alpha + beta,
        'shock_response': alpha,
        'volatility_persistence': beta,
        'rv_impact': delta,
        't_stat_delta': t_stats[4],
        'delta_significant': abs(t_stats[4]) > 1.96,
        'delta_positive': delta > 0
    }
    
    return interpretation
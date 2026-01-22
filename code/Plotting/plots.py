import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf as statsmodels_plot_acf, plot_pacf as statsmodels_plot_pacf
from matplotlib.dates import DateFormatter
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_histogram(data, title, xlabel, ylabel, save_path, mean=None):
    """
    Plot a histogram of the data with normal curve overlay for comparison.
    
    Parameters:
    - data: array-like, data to plot
    - title: str, plot title
    - xlabel: str, x-axis label
    - ylabel: str, y-axis label
    - save_path: str, path to save the figure
    - mean: float, optional mean value to plot as vertical line
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram using seaborn for better aesthetics
    sns.histplot(data, bins=100, stat='density', alpha=0.7, label='Histogram', 
                 color='steelblue', edgecolor='black', linewidth=0.5, ax=ax)
    
    # Calculate normal distribution overlay for comparison
    mu = np.mean(data)
    sigma = np.std(data)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    ax.plot(x, norm.pdf(x, mu, sigma), 'k--', linewidth=2, label='Normal distribution')
    
    # Add mean line if provided
    if mean is not None:
        ax.axvline(mean, linestyle='--', linewidth=2, color='red', label=f'Mean = {mean:.2f}')
    
    # Set labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_acf(data, title, xlabel, ylabel, save_path, lags=20):
    """
    Plot ACF (Autocorrelation Function) of the data.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    statsmodels_plot_acf(data, lags=lags, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pacf(data, title, xlabel, ylabel, save_path, lags=20):
    """
    Plot PACF (Partial Autocorrelation Function) of the data.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    statsmodels_plot_pacf(data, lags=lags, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_timeseries(data, title, xlabel, ylabel, save_path, label=None, x_data=None):
    """
    Plot time series data as a line plot.
    
    Parameters:
    - data: array-like, time series data to plot
    - title: str, plot title
    - xlabel: str, x-axis label
    - ylabel: str, y-axis label
    - save_path: str, path to save the figure
    - label: str, optional label for the legend
    - x_data: array-like, optional x-axis data (if None, uses index)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the time series
    use_dates = False
    if x_data is not None:
        # Check if x_data contains datetime objects (pandas Series or numpy datetime array)
        if isinstance(x_data, pd.Series):
            use_dates = pd.api.types.is_datetime64_any_dtype(x_data)
        elif hasattr(x_data, 'dtype'):
            use_dates = pd.api.types.is_datetime64_any_dtype(x_data) or np.issubdtype(x_data.dtype, np.datetime64)
        
        ax.plot(x_data, data, label=label, linewidth=1.5)
        
        # Format dates on x-axis if datetime
        if use_dates:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.plot(data, label=label, linewidth=1.5)
    
    # Set labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add legend if label is provided
    if label is not None:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiple_timeseries(data_list, labels_list, title, xlabel, ylabel, save_path, x_data=None, colors=None):
    """
    Plot multiple time series on the same figure for comparison.
    
    Parameters:
    - data_list: list of array-like, time series data to plot (each element is one series)
    - labels_list: list of str, labels for each series (for legend)
    - title: str, plot title
    - xlabel: str, x-axis label
    - ylabel: str, y-axis label
    - save_path: str, path to save the figure
    - x_data: array-like, optional x-axis data (if None, uses index)
    - colors: list of str, optional colors for each series (defaults to contrasting colors)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Default contrasting colors: bright blue and deep orange/red for high contrast
    if colors is None:
        # First two are highly contrasting: deep blue and bright orange
        # Additional colors for more series
        colors = ['#0066CC', '#FF6600', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if x_data contains datetime objects
    use_dates = False
    if x_data is not None:
        if isinstance(x_data, pd.Series):
            use_dates = pd.api.types.is_datetime64_any_dtype(x_data)
        elif hasattr(x_data, 'dtype'):
            use_dates = pd.api.types.is_datetime64_any_dtype(x_data) or np.issubdtype(x_data.dtype, np.datetime64)
    
    # Plot each time series with contrasting colors
    for i, (data, label) in enumerate(zip(data_list, labels_list)):
        color = colors[i % len(colors)]  # Cycle through colors if more series than colors
        if x_data is not None:
            ax.plot(x_data, data, label=label, linewidth=2, color=color)
        else:
            ax.plot(data, label=label, linewidth=2, color=color)
    
    # Format dates on x-axis if datetime
    if use_dates:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Set labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
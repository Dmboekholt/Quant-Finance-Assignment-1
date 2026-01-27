"""
Configuration file for machine learning assignment
Contains all paths and configuration settings
"""

import os
import logging
import datetime

#----------------------------------------#
# Paths
#----------------------------------------#
# Base directory (assignment root, one level up from code folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Background files path
ASSIGNMENT_FILES_DIR = os.path.join(BASE_DIR, 'assignment')

# Plots path
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Log Assignments path
LOG_ASSIGNMENTS_DIR = os.path.join(BASE_DIR, 'log_assignment')


#----------------------------------------#
# Logging Configuration
#----------------------------------------#
# Toggle logging on/off
LOG_ENABLED = True  # Set to False to disable logging

def setup_logging():
    """
    Setup logging configuration with timestamped log files.
    Creates a new log file for each run with format: run_YYYYMMDD_HHMMSS.log
    """
    if not LOG_ENABLED:
        # Disable logging by setting level to a high value
        logging.basicConfig(
            level=logging.CRITICAL + 1,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return
    
    # Ensure log directory exists
    os.makedirs(LOG_ASSIGNMENTS_DIR, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp}.log"
    log_filepath = os.path.join(LOG_ASSIGNMENTS_DIR, log_filename)

    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_filepath}")

import os
import logging
import yaml
import pandas as pd
from typing import Dict, Any, Optional

# =============================================================================
# === 1. SYSTEM UTILITIES ===
# =============================================================================

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures a standardized logger for the application.
    
    Args:
        name (str): The name of the module calling the logger (usually __name__).
        log_file (str, optional): If provided, writes logs to this file as well as the console.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if the logger is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            datefmt='%H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler (Optional)
        if log_file:
            # Ensure the logs directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads the YAML configuration file containing dates, targets, and overrides.
    
    Args:
        filepath (str): The relative or absolute path to the settings.yaml file.
        
    Returns:
        Dict[str, Any]: A nested dictionary of the configuration settings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found at: {filepath}")
        
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

# =============================================================================
# === 2. PANDAS DATA & MEMORY UTILITIES ===
# =============================================================================

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names from disparate sources (BigQuery vs. Google Sheets) 
    into a single, unified naming convention for the allocation engine.
    
    Args:
        df (pd.DataFrame): The raw dataframe to be cleaned.
        
    Returns:
        pd.DataFrame: The dataframe with standardized column names.
    """
    col_map = {
        'riskDefinedProductSubscriptionType': 'rpgt', 
        'risk_defined_subscription_product_type': 'rpgt', 
        'RPGT': 'rpgt', 
        'company': 'Company', 
        'Brand': 'Company', 
        'brand': 'Company', 
        'gateway_fid': 'gatewayFid', 
        'Gateway_Src': 'gatewayFid', 
        'Gateway': 'gatewayFid', 
        'paymentmethodprovider': 'paymentMethodProvider', 
        'country': 'Country', 
        'bin': 'BIN', 
        'Bin': 'BIN', 
        'currency': 'Currency', 
        'fcpnumber': 'fcpNumber', 
        'attemptnumber': 'attemptNumber'
    }
    
    rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
        
    # Standardize BINs to drop trailing decimals
    if 'BIN' in df.columns:
        df['BIN'] = df['BIN'].astype(str).str.split('.').str[0].str.strip()
        
    # Ensure attemptNumber exists and is a string
    if 'attemptNumber' in df.columns:
        df['attemptNumber'] = df['attemptNumber'].astype(str).str.lower().str.strip()
    else:
        df['attemptNumber'] = '1'
        
    return df

def clean_key_col(series: pd.Series, remove_dot_zero: bool = False) -> pd.Series:
    """
    Converts memory-heavy text columns into highly compressed Pandas Categories, 
    drastically reducing RAM footprint during massive matrix joins.
    
    Args:
        series (pd.Series): The pandas text column to compress.
        remove_dot_zero (bool): If True, strips '.0' from the end of numeric strings.
        
    Returns:
        pd.Series: A memory-optimized Categorical series.
    """
    cat_series = series.astype('category')
    
    if remove_dot_zero:
        cat_map = {
            c: str(c).lower().strip()[:-2] if str(c).lower().strip().endswith('.0') else str(c).lower().strip() 
            for c in cat_series.cat.categories
        }
    else:
        cat_map = {c: str(c).lower().strip() for c in cat_series.cat.categories}
        
    return cat_series.map(cat_map).astype('category')
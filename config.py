# config.py
"""
Configuration file for TFT model training and forecasting
"""

# Default configuration
DEFAULT_CONFIG = {
    "data": {
        "synthetic": True,        # Whether to use synthetic data (True) or real data (False)
        "file_path": None,        # Path to real data file (CSV)
        "start_date": "2022-01-01",  # For synthetic data
        "end_date": "2023-12-31",    # For synthetic data
        "num_skus": 5             # Number of SKUs for synthetic data
    },
    "model": {
        "max_encoder_length": 90,    # Maximum encoder length (lookback) - Increased from 60 to 90
        "max_prediction_length": 35,  # Forecast horizon (looking forward)
        "hidden_size": 128,           # Hidden dimension in TFT - Increased from 32 to 128
        "attention_head_size": 4,    # Number of attention heads - Increased from 1 to 4
        "dropout": 0.2,              # Dropout rate - Increased from 0.1 to 0.2
        "hidden_continuous_size": 64, # Hidden continuous size - Increased from 16 to 64
        "learning_rate": 0.001        # Learning rate for optimizer - Decreased from 0.01 to 0.001
    },
    "training": {
        "batch_size": 32,            # Batch size for training - Decreased from 64 to 32
        "num_epochs": 50,            # Maximum number of epochs - Increased from 10 to 50
        "early_stopping_patience": 10, # Early stopping patience - Increased from 3 to 10
        "log_interval": 10           # Log interval for batches
    },
    "output": {
        "forecasts_dir": "./forecasts", # Directory for forecast outputs
        "plots_dir": "./plots",         # Directory for plots
        "models_dir": "./models"        # Directory for saved models
    }
}

# Configuration for year-round items
YEAR_ROUND_CONFIG = {
    "data": DEFAULT_CONFIG["data"],
    "model": {
        "max_encoder_length": 90,
        "max_prediction_length": 35,
        "hidden_size": 160,            # Increased from 64 to 160
        "attention_head_size": 4,     
        "dropout": 0.2,
        "hidden_continuous_size": 64,  # Increased from 32 to 64
        "learning_rate": 0.001        
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 50,
        "early_stopping_patience": 10,
        "log_interval": 10
    },
    "output": DEFAULT_CONFIG["output"]
}

# Configuration for highly seasonal items
SEASONAL_CONFIG = {
    "data": DEFAULT_CONFIG["data"],
    "model": {
        "max_encoder_length": 120,      # Longer lookback for seasonal patterns - Increased from 90 to 120
        "max_prediction_length": 35,
        "hidden_size": 160,            # Increased from 128 to 160
        "attention_head_size": 4,
        "dropout": 0.2,               
        "hidden_continuous_size": 64,
        "learning_rate": 0.001
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 100,             # Increased from 50 to 100
        "early_stopping_patience": 15, # Increased from 10 to 15
        "log_interval": 5
    },
    "output": DEFAULT_CONFIG["output"]
}

# Configuration for semi-seasonal items
SEMI_SEASONAL_CONFIG = {
    "data": DEFAULT_CONFIG["data"],
    "model": {
        "max_encoder_length": 90,
        "max_prediction_length": 35,
        "hidden_size": 128,
        "attention_head_size": 4,
        "dropout": 0.2,
        "hidden_continuous_size": 64,
        "learning_rate": 0.001
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 75,
        "early_stopping_patience": 10,
        "log_interval": 8
    },
    "output": DEFAULT_CONFIG["output"]
}

# Configuration for new SKUs
NEW_SKU_CONFIG = {
    "data": DEFAULT_CONFIG["data"],
    "model": {
        "max_encoder_length": 30,      # Short lookback (limited history)
        "max_prediction_length": 35,
        "hidden_size": 64,             # Smaller size for limited data
        "attention_head_size": 2,
        "dropout": 0.1,
        "hidden_continuous_size": 32,
        "learning_rate": 0.005         # Higher learning rate to learn quickly
    },
    "training": {
        "batch_size": 16,              # Smaller batch size
        "num_epochs": 30,              # Fewer epochs (limited data)
        "early_stopping_patience": 5,
        "log_interval": 5
    },
    "output": DEFAULT_CONFIG["output"]
}

# Function to get appropriate config based on SKU segment
def get_config_by_segment(segment):
    """Get configuration based on SKU segment"""
    if segment == "year_round":
        return YEAR_ROUND_CONFIG
    elif segment == "seasonal":
        return SEASONAL_CONFIG
    elif segment == "semi_seasonal":
        return SEMI_SEASONAL_CONFIG
    elif segment == "new_sku":
        return NEW_SKU_CONFIG
    else:
        return DEFAULT_CONFIG
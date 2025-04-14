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
        "max_encoder_length": 60,    # Maximum encoder length (lookback)
        "max_prediction_length": 35,  # Forecast horizon (looking forward)
        "hidden_size": 32,           # Hidden dimension in TFT
        "attention_head_size": 1,    # Number of attention heads
        "dropout": 0.1,              # Dropout rate
        "hidden_continuous_size": 16, # Hidden continuous size
        "learning_rate": 0.01        # Learning rate for optimizer
    },
    "training": {
        "batch_size": 64,            # Batch size for training
        "num_epochs": 10,            # Maximum number of epochs
        "early_stopping_patience": 3, # Early stopping patience
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
        "max_encoder_length": 60,
        "max_prediction_length": 35,
        "hidden_size": 64,            # Increased for complex patterns
        "attention_head_size": 4,     # Multiple heads for better attention
        "dropout": 0.1,
        "hidden_continuous_size": 32,  # Larger for more capacity
        "learning_rate": 0.001        # Lower learning rate for stability
    },
    "training": {
        "batch_size": 64,
        "num_epochs": 50,             # More epochs for convergence
        "early_stopping_patience": 5,  # More patience
        "log_interval": 10
    },
    "output": DEFAULT_CONFIG["output"]
}

# Configuration for highly seasonal items
SEASONAL_CONFIG = {
    "data": DEFAULT_CONFIG["data"],
    "model": {
        "max_encoder_length": 90,      # Longer lookback for seasonal patterns
        "max_prediction_length": 35,
        "hidden_size": 128,            # Much larger for capturing complex seasonality
        "attention_head_size": 4,
        "dropout": 0.2,                # Higher dropout to prevent overfitting
        "hidden_continuous_size": 64,
        "learning_rate": 0.001
    },
    "training": {
        "batch_size": 32,              # Smaller batch size for detailed learning
        "num_epochs": 100,             # Many epochs for seasonal learning
        "early_stopping_patience": 10, # Much more patience
        "log_interval": 5
    },
    "output": DEFAULT_CONFIG["output"]
}

# Configuration for semi-seasonal items
SEMI_SEASONAL_CONFIG = {
    "data": DEFAULT_CONFIG["data"],
    "model": {
        "max_encoder_length": 75,      # Medium lookback
        "max_prediction_length": 35,
        "hidden_size": 96,             # Balanced size
        "attention_head_size": 3,      # Balanced attention
        "dropout": 0.15,               # Medium dropout
        "hidden_continuous_size": 48,
        "learning_rate": 0.002
    },
    "training": {
        "batch_size": 48,
        "num_epochs": 75,
        "early_stopping_patience": 7,
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
        "hidden_size": 48,             # Medium size
        "attention_head_size": 2,
        "dropout": 0.1,
        "hidden_continuous_size": 24,
        "learning_rate": 0.005         # Higher learning rate to learn quickly
    },
    "training": {
        "batch_size": 32,
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
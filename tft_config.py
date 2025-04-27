# Base configuration for TFT model
BASE_CONFIG = {
    "max_prediction_length": 35,  # 35-day forecast horizon
    "max_encoder_length": 180,    # 6 months of history
    "max_epochs": 50,            # Increased epochs
    "learning_rate": 0.01,       # Reduced learning rate
    "hidden_size": 32,           # Increased hidden size
    "attention_head_size": 4,    # Increased attention heads
    "dropout": 0.2,             # Increased dropout
    "hidden_continuous_size": 16, # Increased continuous size
    "allow_missing_timesteps": True,  # Allow missing timesteps in the data
    "special_days": [
        "holiday",
        "promotion",
        "special_event"
    ],
    "static_reals": [
        "base_price",
        "avg_historical_volume"
    ],
    "time_varying_known_categoricals": [
        "month",
        "day_of_week",
        "is_holiday",
        "is_promotion"
    ],
    "time_varying_known_reals": [
        "time_idx",
        "price",
        "promotion_depth",
        "days_until_holiday",
        "days_since_holiday",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos"
    ]
}

# Segment-specific configurations
SEGMENT_CONFIGS = {
    "year_round": {
        **BASE_CONFIG,
        "hidden_size": 64,        # Larger model for complex patterns
        "attention_head_size": 8,
        "dropout": 0.3,
        "max_encoder_length": 365  # Full year history
    },
    "highly_seasonal": {
        **BASE_CONFIG,
        "hidden_size": 48,
        "attention_head_size": 6,
        "dropout": 0.25,
        "max_encoder_length": 365  # Full year history for seasonality
    },
    "semi_seasonal": {
        **BASE_CONFIG,
        "hidden_size": 40,
        "attention_head_size": 4,
        "dropout": 0.25,
        "max_encoder_length": 180  # 6 months history
    },
    "new_sku": {
        **BASE_CONFIG,
        "hidden_size": 32,
        "attention_head_size": 4,
        "dropout": 0.2,
        "max_encoder_length": 90,  # 3 months history
        "max_epochs": 30  # Fewer epochs for faster training
    }
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 256,           # Increased batch size
    "early_stopping_patience": 15,
    "gradient_clip_val": 0.1,
    "optimizer": "ranger",
    "reduce_on_plateau_patience": 6,
    "num_workers": 0,            # Increased workers
    "limit_train_batches": 100,  # Increased training batches
    "log_interval": 10,
    "accumulate_grad_batches": 2  # Gradient accumulation
}

# Hyperparameter optimization configuration
HYPERPARAM_OPT_CONFIG = {
    "n_trials": 20,              # Increased trials
    "gradient_clip_val_range": (0.01, 1.0),
    "hidden_size_range": (16, 128),
    "hidden_continuous_size_range": (16, 128),
    "attention_head_size_range": (2, 8),
    "learning_rate_range": (0.001, 0.1),
    "dropout_range": (0.1, 0.4),
    "limit_train_batches": 50,
    "reduce_on_plateau_patience": 6
} 
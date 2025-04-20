# Base configuration for TFT model
BASE_CONFIG = {
    "max_prediction_length": 35,  # 35-day forecast horizon
    "max_encoder_length": 90,     # 3 months of history
    "max_epochs": 3,
    "learning_rate": 0.03,
    "hidden_size": 16,
    "attention_head_size": 2,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
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
        "days_since_holiday"
    ]
}

# Segment-specific configurations
SEGMENT_CONFIGS = {
    "year_round": {
        **BASE_CONFIG,
        "hidden_size": 32,  # Larger model for complex patterns
        "attention_head_size": 4,
        "dropout": 0.2,
        "max_encoder_length": 120  # Longer history for stable patterns
    }
    # ,
    # "highly_seasonal": {
    #     **BASE_CONFIG,
    #     "hidden_size": 24,
    #     "attention_head_size": 3,
    #     "dropout": 0.15,
    #     "max_encoder_length": 365  # Full year history for seasonality
    # },
    # "semi_seasonal": {
    #     **BASE_CONFIG,
    #     "hidden_size": 20,
    #     "attention_head_size": 2,
    #     "dropout": 0.15,
    #     "max_encoder_length": 180  # 6 months history
    # },
    # "new_sku": {
    #     **BASE_CONFIG,
    #     "hidden_size": 16,
    #     "attention_head_size": 2,
    #     "dropout": 0.1,
    #     "max_encoder_length": 60,  # Limited history
    #     "max_epochs": 2  # Fewer epochs for faster training
    # }
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 128,
    "early_stopping_patience": 10,
    "gradient_clip_val": 0.1,
    "optimizer": "ranger",
    "reduce_on_plateau_patience": 4,
    "num_workers": 0,
    "limit_train_batches": 50,
    "log_interval": 10
}

# Hyperparameter optimization configuration
HYPERPARAM_OPT_CONFIG = {
    "n_trials": 1,
    "gradient_clip_val_range": (0.01, 1.0),
    "hidden_size_range": (8, 128),
    "hidden_continuous_size_range": (8, 128),
    "attention_head_size_range": (1, 4),
    "learning_rate_range": (0.001, 0.1),
    "dropout_range": (0.1, 0.3),
    "limit_train_batches": 30,
    "reduce_on_plateau_patience": 4
} 
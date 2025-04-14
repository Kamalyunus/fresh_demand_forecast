import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

def create_training_dataset(training_data, max_encoder_length=90, max_prediction_length=35):
    """Create TimeSeriesDataSet with enhanced features"""
    # Get all time-varying real features that are not the target
    time_varying_known_reals = [
        "time_idx", "month_sin", "month_cos", 
        "day_of_week_sin", "day_of_week_cos",
        "days_to_holiday", "days_since_holiday"
    ]
    
    # Add lag and rolling features
    for col in training_data.columns:
        if col.startswith('demand_lag_') or col.startswith('demand_rolling_'):
            time_varying_known_reals.append(col)
    
    # Add cross-product features if they exist
    for col in training_data.columns:
        if col.startswith('avg_related_') or col.startswith('num_related_') or col == 'joint_promos':
            time_varying_known_reals.append(col)
            
    # Add category aggregate features if they exist
    for col in training_data.columns:
        if col.startswith('category_'):
            time_varying_known_reals.append(col)
            
    # Make sure none of our features have NA values
    for col in time_varying_known_reals:
        if col in training_data.columns and training_data[col].isna().any():
            print(f"Warning: Column {col} has NA values. Filling with zeros.")
            training_data[col] = training_data[col].fillna(0)
    
    print(f"Creating TimeSeriesDataSet with time-varying known reals: {time_varying_known_reals}")
    
    training = TimeSeriesDataSet(
        data=training_data,
        time_idx="time_idx",
        target="demand",
        group_ids=["sku"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["sku"],
        time_varying_known_categoricals=["month", "day_of_week", "is_holiday"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=["demand"],
        target_normalizer=GroupNormalizer(
            groups=["sku"], transformation="softplus", center=False
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    return training

def create_dataloaders(training_dataset, batch_size=32):
    """Create train and validation dataloaders"""
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = training_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    
    return train_dataloader, val_dataloader

def create_tft_model(training_dataset, learning_rate=0.001, hidden_size=128, 
                    attention_head_size=4, dropout=0.2, hidden_continuous_size=64):
    """Create Temporal Fusion Transformer model with improved parameters"""
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,                    # Increased from 32 to 128
        attention_head_size=attention_head_size,    # Increased from 1 to 4
        dropout=dropout,                            # Increased from 0.1 to 0.2
        hidden_continuous_size=hidden_continuous_size,  # Increased from 16 to 64
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=5,               # Added patience for learning rate scheduling
        logging_metrics=torch.nn.ModuleList([]),    # Avoid excessive logging
    )
    
    return model
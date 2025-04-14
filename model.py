import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

def create_training_dataset(training_data, max_encoder_length=60, max_prediction_length=35):
    """Create TimeSeriesDataSet for training"""
    training = TimeSeriesDataSet(
        data=training_data,
        time_idx="time_idx",
        target="demand",
        group_ids=["sku"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["sku"],
        time_varying_known_categoricals=["month", "day_of_week"],
        time_varying_known_reals=[
            "time_idx", "month_sin", "month_cos", 
            "day_of_week_sin", "day_of_week_cos"
        ],
        time_varying_unknown_reals=["demand"],
        target_normalizer=GroupNormalizer(
            groups=["sku"], transformation="softplus", center=False
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    return training

def create_dataloaders(training_dataset, batch_size=64):
    """Create train and validation dataloaders"""
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = training_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    
    return train_dataloader, val_dataloader

def create_tft_model(training_dataset, learning_rate=0.01, hidden_size=32, 
                    attention_head_size=1, dropout=0.1, hidden_continuous_size=16):
    """Create Temporal Fusion Transformer model"""
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
    )
    
    return model
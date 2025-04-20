import copy
from pathlib import Path
import warnings
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import (
    TemporalFusionTransformer, 
    TimeSeriesDataSet,
    GroupNormalizer,
    MAE, 
    SMAPE, 
    QuantileLoss
)
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
torch.set_float32_matmul_precision('high')


class TFTForecaster:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.best_model = None
        self.training_dataset = None
        self.validation_dataset = None
        
    def prepare_data(self, data):
        """Prepare data for TFT model"""
        # Add time index if not present
        if "time_idx" not in data.columns:
            data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
            data["time_idx"] -= data["time_idx"].min()
        
        # Add additional features
        data["month"] = data.date.dt.month.astype(str).astype("category")
        data["log_volume"] = np.log(data.volume + 1e-8)
        
        # Add segment-specific features
        data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
        data["avg_volume_by_segment"] = data.groupby(["time_idx", "segment"], observed=True).volume.transform("mean")
        
        # Handle special days
        special_days = self.config.get("special_days", [])
        if special_days:
            for col in special_days:
                if col in data.columns:
                    data[col] = data[col].map({0: "-", 1: col}).astype("category")
                else:
                    data[col] = "-"
        
        return data
    
    def create_datasets(self, data):
        """Create training and validation datasets"""
        max_prediction_length = self.config["max_prediction_length"]
        max_encoder_length = self.config["max_encoder_length"]
        
        # Ensure date column is datetime
        data["date"] = pd.to_datetime(data["date"])
        
        # Create time index if not present
        if "time_idx" not in data.columns:
            data["time_idx"] = data["date"].dt.year * 365 + data["date"].dt.dayofyear
            data["time_idx"] -= data["time_idx"].min()
        
        # Convert categorical columns to string type
        categorical_columns = ["sku", "segment", "month", "holiday", "promotion", "special_event"]
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype(str)
        
        # Ensure all required columns exist
        required_columns = [
            "time_idx", "volume", "sku", "segment", "month", "price",
            "holiday", "promotion", "special_event"
        ]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Set index to avoid issues with TimeSeriesDataSet
        data = data.set_index(["time_idx", "sku"]).sort_index()
        
        training_cutoff = data.index.get_level_values("time_idx").max() - max_prediction_length
        
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            data.reset_index(),
            time_idx="time_idx",
            target="volume",
            group_ids=["sku"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["sku", "segment"],
            static_reals=self.config.get("static_reals", []),
            time_varying_known_categoricals=self.config.get("time_varying_known_categoricals", ["month"]),
            time_varying_known_reals=self.config.get("time_varying_known_reals", ["time_idx", "price"]),
            time_varying_unknown_categoricals=["holiday", "promotion", "special_event"],
            time_varying_unknown_reals=["volume"],
            target_normalizer=GroupNormalizer(
                groups=["sku"], 
                transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            data.reset_index(), 
            predict=True, 
            stop_randomization=True
        )
        
        return self.training_dataset, self.validation_dataset
    
    def create_dataloaders(self, batch_size=128):
        """Create training and validation dataloaders"""
        train_dataloader = self.training_dataset.to_dataloader(
            train=True, 
            batch_size=batch_size, 
            num_workers=0
        )
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False, 
            batch_size=batch_size, 
            num_workers=0
        )
        return train_dataloader, val_dataloader
    
    def train(self, train_dataloader, val_dataloader):
        """Train the TFT model"""
        # Configure callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode="min"
        )
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")
        
        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            devices=1,
            accelerator="gpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )
        
        # Initialize model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.config["learning_rate"],
            hidden_size=self.config["hidden_size"],
            attention_head_size=self.config["attention_head_size"],
            dropout=self.config["dropout"],
            hidden_continuous_size=self.config["hidden_continuous_size"],
            loss=QuantileLoss(),
            log_interval=10,
            optimizer="ranger",
            reduce_on_plateau_patience=4,
        )
        
        # Train model
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Save best model
        self.best_model = TemporalFusionTransformer.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        
        return self.best_model, trainer
    
    def optimize_hyperparameters(self, train_dataloader, val_dataloader, n_trials=2):
        """Optimize hyperparameters using Optuna"""
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path="optuna_test",
            n_trials=n_trials,
            max_epochs=self.config["max_epochs"],
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30, devices=1),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )
        
        # Save study results
        with open("tft_study.pkl", "wb") as fout:
            pickle.dump(study, fout)
            
        return study.best_trial.params
    
    def predict(self, dataloader, mode="prediction", return_y=True, return_x=False):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
            
        predictions = self.best_model.predict(
            dataloader,
            mode=mode,
            return_y=return_y,
            return_x=return_x,
            trainer_kwargs=dict(accelerator="gpu", devices=1)
    )
        return predictions
    
    def evaluate(self, predictions):
        """Evaluate model performance"""
        mae = MAE()(predictions.output, predictions.y)
        smape = SMAPE()(predictions.output, predictions.y)
        return {"MAE": mae, "SMAPE": smape}
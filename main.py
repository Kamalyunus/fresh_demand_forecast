import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tft_forecast import TFTForecaster
from tft_config import SEGMENT_CONFIGS, TRAINING_CONFIG, HYPERPARAM_OPT_CONFIG
from data_generator import create_synthetic_data
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load and preprocess the data"""
    create_synthetic_data()
    logger.info("Sample data generated successfully!")

    logger.info("Loading data...")    
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    # Ensure numeric columns are float
    numeric_columns = ["volume", "price", "base_price", "promotion_depth"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].astype(float)
    
    # Ensure all required columns exist
    required_columns = [
        "time_idx", "volume", "sku", "segment", "month", "price",
        "holiday", "promotion", "special_event", "day_of_week"
    ]
    
    # Add time index if not present
    if "time_idx" not in data.columns:
        data["time_idx"] = data["date"].dt.year * 365 + data["date"].dt.dayofyear
        data["time_idx"] -= data["time_idx"].min()
    
    # Convert categorical columns to string type
    categorical_columns = [
        "sku", "segment", "month", "holiday", "promotion", 
        "special_event", "day_of_week", "is_holiday", "is_promotion"
    ]
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype(str)
    
    # Fill any missing columns with default values
    for col in required_columns:
        if col not in data.columns:
            if col == "holiday":
                data[col] = "0"
            elif col == "promotion":
                data[col] = "0"
            elif col == "special_event":
                data[col] = "0"
            elif col == "price":
                data[col] = data.get("base_price", 10.0).astype(float)
            elif col == "day_of_week":
                data[col] = data["date"].dt.dayofweek.astype(str)
    
    return data

def segment_skus(data):
    """Segment SKUs based on sales patterns"""
    logger.info("Segmenting SKUs...")
    
    # If segment is already in the data, use it
    if 'segment' in data.columns:
        return data
    
    # Calculate metrics for segmentation
    sku_metrics = data.groupby('sku').agg({
        'volume': [
            ('total_days', 'count'),
            ('zero_days', lambda x: (x == 0).sum()),
            ('cv', lambda x: x.std() / x.mean() if x.mean() > 0 else np.inf)
        ]
    }).reset_index()
    
    sku_metrics.columns = ['sku', 'total_days', 'zero_days', 'cv']
    
    # Assign segments
    sku_metrics['segment'] = 'new_sku'  # Default segment
    
    # Year-round items: Sales in ≥90% of days
    year_round_mask = (sku_metrics['total_days'] >= 0.9 * data['date'].nunique())
    sku_metrics.loc[year_round_mask, 'segment'] = 'year_round'
    
    # Highly seasonal items: Zero sales for ≥60% of year
    seasonal_mask = (sku_metrics['zero_days'] / sku_metrics['total_days'] >= 0.6)
    sku_metrics.loc[seasonal_mask, 'segment'] = 'highly_seasonal'
    
    # Semi-seasonal items: High variation (>100% coefficient)
    semi_seasonal_mask = (sku_metrics['cv'] > 1.0) & ~year_round_mask & ~seasonal_mask
    sku_metrics.loc[semi_seasonal_mask, 'segment'] = 'semi_seasonal'
    
    # Merge segments back to main data
    data = data.merge(sku_metrics[['sku', 'segment']], on='sku', how='left')
    
    return data

def save_best_model(model, segment, output_dir):
    """Save the best model for a segment"""
    logger.info(f"Saving best model for {segment} segment...")
    
    # Create directory for models
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, f'{segment}_best_model.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

def train_models(data, config_path):
    """Train TFT models for each segment"""
    logger.info("Training models for each segment...")
    models = {}
    
    for segment, config in SEGMENT_CONFIGS.items():
        logger.info(f"Training model for {segment} segment...")
        
        # Filter data for segment
        segment_data = data[data['segment'] == segment].copy()
        if len(segment_data) == 0:
            logger.warning(f"No data found for {segment} segment. Skipping...")
            continue
        
        # Initialize forecaster
        forecaster = TFTForecaster(config)
        
        # Prepare data
        segment_data = forecaster.prepare_data(segment_data)
        
        # Create datasets
        training_dataset, validation_dataset = forecaster.create_datasets(segment_data)
        
        # Create dataloaders
        train_dataloader, val_dataloader = forecaster.create_dataloaders(
            batch_size=TRAINING_CONFIG['batch_size']
        )
        
        # Optimize hyperparameters
        logger.info(f"Optimizing hyperparameters for {segment} segment...")
        best_params = forecaster.optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            n_trials=HYPERPARAM_OPT_CONFIG['n_trials']
        )
        
        # Update config with best parameters
        config.update(best_params)
        logger.info(f"best_param for {segment} segment...{best_params}")

        # Train model
        logger.info(f"Training final model for {segment} segment...")
        model, trainer = forecaster.train(train_dataloader, val_dataloader)
        
        # Save best model
        save_best_model(model, segment, config_path)
        
        # Store model
        models[segment] = {
            'model': model,
            'forecaster': forecaster,
            'config': config
        }
        
    return models

def make_predictions(model_info, output_dir, segment_name):
    """Make predictions using the trained model."""
    try:
        # Get validation dataloader from the forecaster
        val_dataloader = model_info['forecaster'].validation_dataset.to_dataloader(
            train=False,
            batch_size=128,
            num_workers=0
        )

        # Make predictions
        raw_predictions = model_info['forecaster'].predict(
            val_dataloader,
            mode="raw",
            return_x=True
        )
        
        # Save feature importance if available
        if hasattr(model_info['forecaster'], 'feature_importance'):
            importance = model_info['forecaster'].feature_importance()
            importance.to_csv(os.path.join(output_dir, f'{segment_name}_feature_importance.csv'))

        # plot 10 examples
        for idx in range(10):  
            fig = model_info['forecaster'].plot_id(raw_predictions, idx=idx)
            fig.savefig(os.path.join(output_dir, f'{segment_name}_prediction_example_{idx}.png'))
        
        return None
        
    except Exception as e:
        logger.error(f"Error making predictions for {segment_name} segment: {str(e)}")
        return None

def evaluate_models(model_info, segment):
    """Evaluate model performance"""
    logger.info("Evaluating models...")
    results = {}
    # Get validation dataloader from the forecaster
    val_dataloader = model_info['forecaster'].validation_dataset.to_dataloader(
        train=False,
        batch_size=128,
        num_workers=0
    )

    # Make predictions
    predictions = model_info['forecaster'].predict(
        val_dataloader,
        mode="prediction",
        return_y=True
    )

    # Evaluate predictions
    metrics = model_info['forecaster'].evaluate(predictions)
    results[segment] = metrics
        
    logger.info(f"{segment} segment metrics: {metrics}")
        
    return results

def main(config=None):
    """Main function to orchestrate the entire workflow"""
    # Default configuration
    if config is None:
        config = {
            "data": {
                "file_path": "data.csv",
                "output_dir": "output"
            }
        }
    
    # Create output directory
    os.makedirs(config["data"]["output_dir"], exist_ok=True)
    
    # Load data
    data = load_data(config["data"]["file_path"])
    
    # Segment SKUs
    data = segment_skus(data)
    
    # Train models
    models = train_models(data, config["data"]["output_dir"])

    # Evaluate models
    results = {}
    for segment, model_info in models.items():
        results[segment] = evaluate_models(model_info, segment)  
    
    # Make predictions
    for segment, model_info in models.items():
         make_predictions(model_info, config["data"]["output_dir"], segment)
    
    logger.info("Forecasting pipeline completed successfully!")

if __name__ == "__main__":
    main() 
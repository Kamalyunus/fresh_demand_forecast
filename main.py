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
    logger.info("Loading data...")
    
    # Check if data file exists, if not generate sample data
    if not os.path.exists(data_path):
        logger.info("Data file not found. Generating sample data...")
        create_synthetic_data()
        logger.info("Sample data generated successfully!")
    
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

def save_training_progress(trainer, segment, output_dir):
    """Save training progress chart for a segment"""
    logger.info(f"Saving training progress for {segment} segment...")
    
    # Create directory for training progress
    progress_dir = os.path.join(output_dir, "training_progress")
    os.makedirs(progress_dir, exist_ok=True)
    
    # Get training metrics
    metrics = pd.DataFrame({
        'epoch': range(len(trainer.callback_metrics)),
        'train_loss': [trainer.callback_metrics.get('train_loss_epoch', 0) for _ in range(len(trainer.callback_metrics))],
        'val_loss': [trainer.callback_metrics.get('val_loss', 0) for _ in range(len(trainer.callback_metrics))]
    })
    
    # Plot training metrics
    plt.figure(figsize=(12, 6))
    for metric in ['train_loss', 'val_loss']:
        if metric in metrics.columns:
            plt.plot(metrics['epoch'], metrics[metric], label=metric)
    
    plt.title(f'Training Progress - {segment} Segment')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(progress_dir, f'{segment}_training_progress.png'))
    plt.close()

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
        
        # Train model
        logger.info(f"Training final model for {segment} segment...")
        model, trainer = forecaster.train(train_dataloader, val_dataloader)
        
        # Save training progress
        save_training_progress(trainer, segment, config_path)
        
        # Save best model
        save_best_model(model, segment, config_path)
        
        # Store model
        models[segment] = {
            'model': model,
            'forecaster': forecaster,
            'config': config
        }
        
    return models

def make_predictions(model, model_info, output_dir, segment_name):
    """Make predictions using the trained model."""
    try:
        # Get validation dataloader from the forecaster
        val_dataloader = model_info['forecaster'].validation_dataset.to_dataloader(
            train=False,
            batch_size=128,
            num_workers=0
        )

        # Make predictions
        pred_output = model_info['forecaster'].predict(
            val_dataloader,
            mode="prediction",
            return_y=True,
            return_x=True
        )

        # Extract predictions and actual values from the output
        predictions_tensor = pred_output[0]  # First element of tuple is predictions
        actual_values = pred_output[1]  # Second element is actual values (x)

        # Move tensors to CPU and convert to numpy
        predictions_np = predictions_tensor.cpu().numpy()
        
        # Create DataFrame with predictions
        n_samples, n_timepoints = predictions_np.shape
        predictions = pd.DataFrame({
            'sample_id': np.repeat(range(n_samples), n_timepoints),
            'time_point': np.tile(range(n_timepoints), n_samples),
            'prediction': predictions_np.flatten()
        })

        if actual_values is not None:
            actual_np = actual_values[0].cpu().numpy()  # First element of tuple contains values
            predictions['actual'] = actual_np.flatten()
        
        # Store predictions by SKU
        predictions_by_sku = {}
        group_ids = model_info['forecaster'].validation_dataset.group_ids
        for i, sku in enumerate(group_ids):
            sku_mask = predictions['sample_id'] == i
            predictions_by_sku[sku] = predictions[sku_mask]
        
        # Save feature importance if available
        if hasattr(model, 'feature_importance'):
            importance = model.feature_importance()
            importance.to_csv(os.path.join(output_dir, f'{segment_name}_feature_importance.csv'))
        
        # Plot predictions for a few examples
        for i in range(min(3, len(group_ids))):
            plt.figure(figsize=(12, 6))
            sku = group_ids[i]
            sku_data = predictions_by_sku[sku]
            
            plt.plot(sku_data['time_point'], sku_data['prediction'], 
                    label='Predicted', linestyle='-')
            if 'actual' in sku_data.columns:
                plt.plot(sku_data['time_point'], sku_data['actual'], 
                        label='Actual', linestyle='--')
            
            plt.title(f'Predictions vs Actual - {sku} ({segment_name})')
            plt.xlabel('Time Point')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'{segment_name}_prediction_example_{i}.png'))
            plt.close()
        
        return predictions_by_sku
        
    except Exception as e:
        logger.error(f"Error making predictions for {segment_name} segment: {str(e)}")
        return None

def evaluate_models(models, predictions):
    """Evaluate model performance"""
    logger.info("Evaluating models...")
    results = {}
    
    for segment, model_info in models.items():
        if segment not in predictions:
            continue
            
        # Evaluate predictions
        metrics = model_info['forecaster'].evaluate(predictions[segment])
        results[segment] = metrics
        
        logger.info(f"{segment} segment metrics: {metrics}")
        
    return results

def create_visualizations(data, predictions, output_dir):
    """Create visualizations for representative SKUs from each segment"""
    logger.info("Creating visualizations...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get representative SKUs from each segment
    segments = data['segment'].unique()
    for segment in segments:
        segment_data = data[data['segment'] == segment]
        skus = segment_data['sku'].unique()
        
        # Select up to 3 SKUs from each segment
        sample_skus = np.random.choice(skus, min(3, len(skus)), replace=False)
        
        for sku in sample_skus:
            sku_data = segment_data[segment_data['sku'] == sku]
            
            # Create time series plot
            plt.figure(figsize=(15, 6))
            plt.plot(sku_data['date'], sku_data['volume'], label='Actual')
            
            # Add predictions if available
            if sku in predictions:
                pred_data = predictions[sku]
                plt.plot(pred_data['date'], pred_data['prediction'], 
                        label='Predicted', linestyle='--')
            
            plt.title(f'Volume over Time - {sku} ({segment})')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(viz_dir, f'{sku}_volume.png'))
            plt.close()
            
            # Create feature importance plot if available
            if f'{segment}_feature_importance' in predictions:
                feature_importance = predictions[f'{segment}_feature_importance']
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance)
                plt.title(f'Feature Importance - {sku} ({segment})')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'{sku}_feature_importance.png'))
                plt.close()
    
    logger.info(f"Visualizations saved to {viz_dir}")

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
    
    # Make predictions
    predictions = {}
    for segment, model_info in models.items():
        predictions[segment] = make_predictions(model_info['model'], model_info, config["data"]["output_dir"], segment)
    
    # Evaluate models
    results = evaluate_models(models, predictions)
    
    # Create visualizations
    create_visualizations(data, predictions, config["data"]["output_dir"])
    
    # Save results
    pd.DataFrame(results).to_csv(
        os.path.join(config["data"]["output_dir"], "model_results.csv")
    )
    
    logger.info("Forecasting pipeline completed successfully!")

if __name__ == "__main__":
    main() 
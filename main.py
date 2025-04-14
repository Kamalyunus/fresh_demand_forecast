# main.py
import os
import warnings
warnings.filterwarnings("ignore")

from data_generator import create_synthetic_data, load_real_data, prepare_data_for_training
from model import create_training_dataset, create_dataloaders, create_tft_model
from training import train_model
from evaluation import evaluate_model
from prediction import generate_forecasts
from visualization import (
    plot_training_history, 
    plot_feature_importance, 
    plot_forecast, 
    plot_all_forecasts,
    plot_seasonal_patterns
)

def main(config=None):
    """Main function to orchestrate the entire workflow"""
    # Default configuration
    if config is None:
        config = {
            "data": {
                "synthetic": True,
                "file_path": None,
                "start_date": "2022-01-01",
                "end_date": "2023-12-31",
                "num_skus": 5
            },
            "model": {
                "max_encoder_length": 60,
                "max_prediction_length": 35,
                "hidden_size": 32,
                "attention_head_size": 1,
                "dropout": 0.1,
                "hidden_continuous_size": 16,
                "learning_rate": 0.01
            },
            "training": {
                "batch_size": 64,
                "num_epochs": 10,
                "early_stopping_patience": 3,
                "log_interval": 10
            },
            "output": {
                "forecasts_dir": "./forecasts",
                "plots_dir": "./plots",
                "models_dir": "./models"
            }
        }
    
    # Create output directories
    os.makedirs(config["output"]["forecasts_dir"], exist_ok=True)
    os.makedirs(config["output"]["plots_dir"], exist_ok=True)
    os.makedirs(config["output"]["models_dir"], exist_ok=True)
    
    # Load or generate data
    print("Step 1: Loading/Generating data...")
    if config["data"]["synthetic"]:
        df = create_synthetic_data(
            start_date=config["data"]["start_date"],
            end_date=config["data"]["end_date"],
            num_skus=config["data"]["num_skus"]
        )
        print(f"Generated synthetic data with {df['sku'].nunique()} SKUs and {len(df)} records")
    else:
        df = load_real_data(config["data"]["file_path"])
        print(f"Loaded real data with {df['sku'].nunique()} SKUs and {len(df)} records")
    
    # Prepare data for training
    print("Step 2: Preparing data for training...")
    training_data, validation_data, training_cutoff = prepare_data_for_training(
        df, 
        max_prediction_length=config["model"]["max_prediction_length"]
    )
    print(f"Training data: {len(training_data)} records")
    print(f"Validation data: {len(validation_data)} records")
    
    # Create dataset
    print("Step 3: Creating dataset...")
    training_dataset = create_training_dataset(
        training_data,
        max_encoder_length=config["model"]["max_encoder_length"],
        max_prediction_length=config["model"]["max_prediction_length"]
    )
    
    # Create dataloaders
    print("Step 4: Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        training_dataset,
        batch_size=config["training"]["batch_size"]
    )
    
    # Create model
    print("Step 5: Creating model...")
    model = create_tft_model(
        training_dataset,
        learning_rate=config["model"]["learning_rate"],
        hidden_size=config["model"]["hidden_size"],
        attention_head_size=config["model"]["attention_head_size"],
        dropout=config["model"]["dropout"],
        hidden_continuous_size=config["model"]["hidden_continuous_size"]
    )
    
    # Attach dataloaders to model for later use
    model.train_dataloader = train_dataloader
    model.val_dataloader = val_dataloader
    
    # Train model
    print("Step 6: Training model...")
    model, history = train_model(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["model"]["learning_rate"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        log_interval=config["training"]["log_interval"]
    )
    
    # Save model
    import torch
    torch.save(model.state_dict(), f"{config['output']['models_dir']}/tft_model.pth")
    print(f"Model saved to {config['output']['models_dir']}/tft_model.pth")
    
    # Plot training history
    print("Step 7: Plotting training history...")
    plot_training_history(history, output_dir=config["output"]["plots_dir"])
    
    # Try to plot feature importance
    try:
        print("Step 8: Plotting feature importance...")
        plot_feature_importance(model, output_dir=config["output"]["plots_dir"])
    except Exception as e:
        print(f"Warning: Could not plot feature importance. Error: {e}")
        print("Continuing with evaluation...")
    
    # Evaluate model
    print("Step 9: Evaluating model...")
    results_df, avg_metrics = evaluate_model(model, val_dataloader, training_dataset, df, training_cutoff)
    results_df.to_csv(f"{config['output']['forecasts_dir']}/evaluation_results.csv", index=False)
    print(f"Evaluation results saved to {config['output']['forecasts_dir']}/evaluation_results.csv")
    
    # Generate forecasts
    print("Step 10: Generating forecasts...")
    forecast_df = generate_forecasts(
        model, 
        training_dataset, 
        df, 
        output_dir=config["output"]["forecasts_dir"],
        training_cutoff=training_cutoff
    )
    
    # Plot forecasts
    print("Step 11: Plotting forecasts...")
    #for sku in df["sku"].unique():
    #    plot_forecast(forecast_df, sku, output_dir=config["output"]["plots_dir"])
        # Plot seasonal patterns
    #     try:
    #         plot_seasonal_patterns(df, sku, output_dir=config["output"]["plots_dir"])
    #     except Exception as e:
    #         print(f"Warning: Could not plot seasonal patterns for {sku}. Error: {e}")
    
    plot_all_forecasts(forecast_df, df, training_cutoff, output_dir=config["output"]["plots_dir"])
    
    print("Process completed successfully!")
    return df, model, forecast_df, results_df

if __name__ == "__main__":
    main()
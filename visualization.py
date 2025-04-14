# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch

def plot_training_history(history, output_dir="./plots"):
    """Plot training and validation losses"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_losses"], label="Training Loss")
    plt.plot(history["val_losses"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/training_history.png")
    print(f"Training history plot saved to {output_dir}/training_history.png")
    plt.close()

def plot_feature_importance(model, output_dir="./plots"):
    """Plot feature importance from TFT model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple example input to generate an output for interpretation
    # We need to generate sample data for interpretation
    device = next(model.parameters()).device
    
    # Create a small batch of data to get model outputs
    # Get an example dataloader
    dummy_loader = model.train_dataloader.loaders[0]  # Use the training dataloader
    x, _ = next(iter(dummy_loader))
    
    # Move to device
    x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    
    # Get an output
    out = model(x)
    
    try:
        # Get feature importance
        interpretation = model.interpret_output(out, reduction="mean")
        
        if 'static_variables' in interpretation:
            # Traditional approach - if static_variables are available
            feature_importance = interpretation['static_variables']
            
            # Convert to list for plotting
            importance_values = []
            labels = []
            for k, v in feature_importance.items():
                labels.append(k)
                importance_values.append(v.cpu().numpy())
        else:
            # Alternative approach - use encoder variables if static vars aren't available
            if 'encoder_variables' in interpretation:
                feature_importance = interpretation['encoder_variables']
                labels = list(feature_importance.keys())
                importance_values = [v.mean().cpu().numpy() for v in feature_importance.values()]
            else:
                # Fallback if no interpretations available
                print("Warning: Could not extract feature importance. Using weights as approximation.")
                # Extract feature weights from embedding layers as approximation
                importance_dict = {}
                for name, param in model.named_parameters():
                    if 'embedding' in name and param.requires_grad:
                        importance_dict[name] = param.abs().mean().item()
                
                labels = list(importance_dict.keys())
                importance_values = list(importance_dict.values())
        
        # Sort by importance
        paired_data = sorted(zip(labels, importance_values), key=lambda x: x[1])
        labels, importance_values = zip(*paired_data)
        
        # Plot
        plt.figure(figsize=(10, max(6, len(labels) * 0.3)))
        plt.barh(labels, importance_values)
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{output_dir}/feature_importance.png")
        print(f"Feature importance plot saved to {output_dir}/feature_importance.png")
        plt.close()
    
    except Exception as e:
        print(f"Warning: Could not plot feature importance. Error: {e}")
        # Create a dummy plot so the process continues
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Feature importance unavailable\nError: {str(e)}", 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance_error.png")
        plt.close()

def plot_forecast(forecast_df, sku, output_dir="./plots"):
    """Plot forecast for a specific SKU"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for the specific SKU
    sku_forecast = forecast_df[forecast_df["sku"] == sku]
    
    if len(sku_forecast) == 0:
        print(f"No forecast data for {sku}")
        return
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot forecast
    plt.plot(sku_forecast["date"], sku_forecast["forecast"], 'r-', label="Forecast")
    
    # Plot prediction interval
    plt.fill_between(
        sku_forecast["date"],
        sku_forecast["lower_bound"],
        sku_forecast["upper_bound"],
        color="red",
        alpha=0.2,
        label="90% Prediction Interval"
    )
    
    # Plot actual values if available
    has_actuals = ~sku_forecast["actual"].isna().all()
    if has_actuals:
        plt.plot(
            sku_forecast["date"],
            sku_forecast["actual"],
            'g-', 
            label="Actual", 
            alpha=0.7
        )
    
    plt.title(f"Demand Forecast for {sku}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/{sku}_forecast.png")
    print(f"Forecast plot for {sku} saved to {output_dir}/{sku}_forecast.png")
    plt.close()

def plot_all_forecasts(forecast_df, df, training_cutoff, output_dir="./plots"):
    """Plot forecasts for all SKUs with historical data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique SKUs
    skus = forecast_df["sku"].unique()
    
    for sku in skus:
        # Filter forecast data for the specific SKU
        sku_forecast = forecast_df[forecast_df["sku"] == sku]
        
        if len(sku_forecast) == 0:
            continue
        
        # Get historical data
        sku_historical = df[df["sku"] == sku].sort_values("date")
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(sku_historical["date"], sku_historical["demand"], 'b-', label="Historical", alpha=0.5)
        
        # Mark the training cutoff
        try:
            cutoff_date = sku_historical[sku_historical["time_idx"] == training_cutoff]["date"].iloc[0]
            plt.axvline(x=cutoff_date, color='k', linestyle='--', label="Forecast Start")
        except:
            # If cannot find exact cutoff, estimate it
            print(f"Warning: Could not find exact cutoff for {sku}, using approximation")
            last_date = sku_historical["date"].max()
            plt.axvline(x=last_date, color='k', linestyle='--', label="Approximate Forecast Start")
        
        # Plot forecast
        plt.plot(sku_forecast["date"], sku_forecast["forecast"], 'r-', label="Forecast")
        
        # Plot prediction interval
        plt.fill_between(
            sku_forecast["date"],
            sku_forecast["lower_bound"],
            sku_forecast["upper_bound"],
            color="red",
            alpha=0.2,
            label="90% Prediction Interval"
        )
        
        # Plot actual values if available
        has_actuals = ~sku_forecast["actual"].isna().all()
        if has_actuals:
            plt.plot(
                sku_forecast["date"],
                sku_forecast["actual"],
                'g-', 
                label="Actual Future", 
                alpha=0.7
            )
        
        plt.title(f"Demand Forecast for {sku}")
        plt.xlabel("Date")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{output_dir}/{sku}_full_forecast.png")
        print(f"Full forecast plot for {sku} saved to {output_dir}/{sku}_full_forecast.png")
        plt.close()

def plot_attention_patterns(model, x, output_dir="./plots"):
    """Plot attention patterns from the TFT model"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get raw attention weights
        raw_attentions = model.interpretation["attention"]
        if raw_attentions is None:
            print("No attention patterns available")
            return
        
        # Get decoder attention weights (if available)
        if "decoder_self_attention" in raw_attentions:
            decoder_attention = raw_attentions["decoder_self_attention"].cpu().numpy()
            
            # Plot decoder attention
            plt.figure(figsize=(10, 8))
            plt.imshow(decoder_attention[0], cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Weight")
            plt.title("Decoder Self-Attention")
            plt.xlabel("Target Position")
            plt.ylabel("Target Position")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/decoder_attention.png")
            plt.close()
        
        # Get encoder-decoder attention (if available)
        if "encoder_decoder_attention" in raw_attentions:
            enc_dec_attention = raw_attentions["encoder_decoder_attention"].cpu().numpy()
            
            # Plot encoder-decoder attention
            plt.figure(figsize=(12, 8))
            plt.imshow(enc_dec_attention[0], cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Weight")
            plt.title("Encoder-Decoder Attention")
            plt.xlabel("Encoder Position")
            plt.ylabel("Decoder Position")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/encoder_decoder_attention.png")
            plt.close()
            
        print(f"Attention pattern plots saved to {output_dir}/")
    
    except Exception as e:
        print(f"Error plotting attention patterns: {e}")

def plot_seasonal_patterns(df, sku, output_dir="./plots"):
    """Plot seasonal patterns for a specific SKU"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for the specific SKU
    sku_data = df[df["sku"] == sku].copy()
    
    if len(sku_data) == 0:
        print(f"No data for {sku}")
        return
    
    # Sort by date
    sku_data = sku_data.sort_values("date")
    
    # Extract month and day of week
    sku_data["month"] = sku_data["date"].dt.month
    sku_data["day_of_week"] = sku_data["date"].dt.dayofweek
    
    # Monthly pattern
    plt.figure(figsize=(12, 6))
    monthly_avg = sku_data.groupby("month")["demand"].mean()
    monthly_std = sku_data.groupby("month")["demand"].std()
    
    plt.plot(monthly_avg.index, monthly_avg.values, 'b-', marker='o')
    plt.fill_between(
        monthly_avg.index,
        monthly_avg - monthly_std,
        monthly_avg + monthly_std,
        color='blue',
        alpha=0.2
    )
    
    plt.title(f"Monthly Demand Pattern for {sku}")
    plt.xlabel("Month")
    plt.ylabel("Average Demand")
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/{sku}_monthly_pattern.png")
    print(f"Monthly pattern plot for {sku} saved to {output_dir}/{sku}_monthly_pattern.png")
    plt.close()
    
    # Day of week pattern
    plt.figure(figsize=(10, 6))
    dow_avg = sku_data.groupby("day_of_week")["demand"].mean()
    dow_std = sku_data.groupby("day_of_week")["demand"].std()
    
    plt.plot(dow_avg.index, dow_avg.values, 'g-', marker='o')
    plt.fill_between(
        dow_avg.index,
        dow_avg - dow_std,
        dow_avg + dow_std,
        color='green',
        alpha=0.2
    )
    
    plt.title(f"Day of Week Demand Pattern for {sku}")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Demand")
    plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/{sku}_dow_pattern.png")
    print(f"Day of week pattern plot for {sku} saved to {output_dir}/{sku}_dow_pattern.png")
    plt.close()
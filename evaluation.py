# evaluation.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_forecasting import TimeSeriesDataSet

def evaluate_model(model, validation_dataloader, training_dataset, df, training_cutoff):
    """Evaluate model performance on validation data"""
    print("Evaluating model...")
    
    # Move model to the right device
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Get parameters
    max_prediction_length = training_dataset.max_prediction_length
    max_encoder_length = training_dataset.max_encoder_length
    
    # Results dictionary
    results = {
        "sku": [],
        "mae": [],
        "rmse": [],
        "mape": []
    }
    
    # For each SKU
    for sku in df["sku"].unique():
        try:
            # Get validation data for this SKU
            sku_data = df[df["sku"] == sku].sort_values("time_idx")
            sku_val_data = sku_data[sku_data["time_idx"] > training_cutoff]
            
            if len(sku_val_data) == 0:
                print(f"No validation data for {sku}")
                continue
                
            actual = sku_val_data["demand"].values[:max_prediction_length]
            
            if len(actual) == 0:
                print(f"No actual values for validation for {sku}")
                continue
            
            # Check if we have enough encoder data
            encoder_data = sku_data[sku_data["time_idx"] <= training_cutoff].copy()
            if len(encoder_data) < max_encoder_length:
                print(f"Warning: Not enough encoder data for {sku}. Available: {len(encoder_data)}, Required: {max_encoder_length}")
                # Use what's available, but must have at least some minimum
                if len(encoder_data) < 5:  # Arbitrary minimum to avoid empty datasets
                    print(f"Skipping {sku} due to insufficient data")
                    continue
            else:
                # Take the last max_encoder_length points
                encoder_data = encoder_data.tail(max_encoder_length)
            
            # Ensure proper categorical types
            encoder_data["sku"] = encoder_data["sku"].astype(str)
            encoder_data["month"] = encoder_data["month"].astype(str)
            encoder_data["day_of_week"] = encoder_data["day_of_week"].astype(str)
            
            # Reset index for prediction
            encoder_data = encoder_data.reset_index(drop=True)
            
            # Make predictions directly using the model
            try:
                # Create a dataset for prediction
                pred_data = TimeSeriesDataSet.from_dataset(
                    training_dataset, 
                    encoder_data, 
                    predict=True, 
                    stop_randomization=True
                )
                pred_loader = pred_data.to_dataloader(batch_size=1, shuffle=False, num_workers=0)
                
                # Generate predictions
                with torch.no_grad():
                    for x, _ in pred_loader:
                        # Move data to device
                        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                        
                        # Get predictions
                        outputs = model(x)
                        if isinstance(outputs, tuple):
                            predictions = outputs[0]
                        else:
                            predictions = outputs
                        
                        # Get prediction values
                        mean_prediction = predictions.mean(1).squeeze().cpu().numpy()
                        
                        # Handle single-value predictions
                        if len(mean_prediction.shape) == 0:
                            mean_prediction = np.array([mean_prediction.item()])
                        
                        # Truncate predictions if needed
                        predicted = mean_prediction[:len(actual)]
                        
                        # Calculate metrics
                        if len(predicted) > 0:
                            mae = mean_absolute_error(actual, predicted)
                            rmse = np.sqrt(mean_squared_error(actual, predicted))
                            
                            # Calculate MAPE - handle zeros
                            mape = np.mean(np.abs((actual - predicted) / np.maximum(1.0, np.abs(actual)))) * 100
                            
                            results["sku"].append(sku)
                            results["mae"].append(mae)
                            results["rmse"].append(rmse)
                            results["mape"].append(mape)
                            
                            print(f"Metrics for {sku}:")
                            print(f"  - MAE: {mae:.4f}")
                            print(f"  - RMSE: {rmse:.4f}")
                            print(f"  - MAPE: {mape:.4f}%")
            except Exception as e:
                print(f"Error evaluating {sku}: {e}")
                # Try an alternative approach using the validation dataloader
                print(f"Using alternative approach for {sku}...")
                try:
                    # Instead of creating a new prediction dataset, use the pre-existing validation dataloader
                    # Filter validation dataloader for this SKU (approximate approach)
                    sku_predictions = []
                    sku_actuals = []
                    
                    # Predict using validation dataloader
                    model.eval()
                    with torch.no_grad():
                        for x, y in validation_dataloader:
                            # Move data to device
                            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                            
                            # Get group ids to filter for this SKU
                            batch_sku_ids = x["groups"][:, 0].cpu().numpy()  # Assuming SKU is the first group
                            
                            # Convert str to int mapping (this is a workaround)
                            id_mapping = {s: i for i, s in enumerate(training_dataset.groupby_ids[0])}
                            sku_id = id_mapping.get(sku)
                            
                            if sku_id is None:
                                continue
                                
                            # Filter for this SKU
                            sku_mask = (batch_sku_ids == sku_id)
                            if not any(sku_mask):
                                continue
                                
                            # Get predictions for this SKU
                            outputs = model(x)
                            if isinstance(outputs, tuple):
                                predictions = outputs[0]
                            else:
                                predictions = outputs
                                
                            # Filter predictions and actuals by SKU
                            sku_pred = predictions[sku_mask].mean(1).cpu().numpy()
                            
                            # Handle y correctly - it's a tuple
                            if isinstance(y, tuple):
                                y_true = y[0]
                            else:
                                y_true = y
                                
                            sku_actual = y_true[sku_mask].cpu().numpy()
                            
                            sku_predictions.append(sku_pred)
                            sku_actuals.append(sku_actual)
                    
                    # Combine predictions and actuals
                    if sku_predictions and sku_actuals:
                        all_predicted = np.concatenate(sku_predictions)
                        all_actual = np.concatenate(sku_actuals)
                        
                        # Calculate metrics
                        mae = mean_absolute_error(all_actual, all_predicted)
                        rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
                        mape = np.mean(np.abs((all_actual - all_predicted) / np.maximum(1.0, np.abs(all_actual)))) * 100
                        
                        results["sku"].append(sku)
                        results["mae"].append(mae)
                        results["rmse"].append(rmse)
                        results["mape"].append(mape)
                        
                        print(f"Alternative metrics for {sku}:")
                        print(f"  - MAE: {mae:.4f}")
                        print(f"  - RMSE: {rmse:.4f}")
                        print(f"  - MAPE: {mape:.4f}%")
                except Exception as inner_e:
                    print(f"Alternative evaluation for {sku} also failed: {inner_e}")
        except Exception as e:
            print(f"Unexpected error evaluating {sku}: {e}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("Warning: No valid evaluation results. Creating dummy results.")
        results_df = pd.DataFrame({
            "sku": ["dummy"],
            "mae": [float('nan')],
            "rmse": [float('nan')],
            "mape": [float('nan')]
        })
        avg_metrics = {
            "Average MAE": float('nan'),
            "Average RMSE": float('nan'),
            "Average MAPE": float('nan')
        }
        return results_df, avg_metrics
    
    # Calculate average metrics
    avg_metrics = {
        "Average MAE": results_df["mae"].mean(),
        "Average RMSE": results_df["rmse"].mean(),
        "Average MAPE": results_df["mape"].mean()
    }
    
    print("Average metrics across all SKUs:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return results_df, avg_metrics
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import warnings
warnings.filterwarnings("ignore")

def create_synthetic_data():
    """Generate synthetic demand data"""
    # Create date range
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    
    # Create SKUs
    skus = [f"SKU_{i}" for i in range(1, 6)]
    
    # Create dataframe
    data = []
    
    # Create time index
    time_idx_map = {date: idx for idx, date in enumerate(dates)}
    
    # Generate data for each SKU
    for sku in skus:
        base_demand = np.random.randint(50, 200)
        
        for date in dates:
            # Simple seasonality and weekday effects
            month_effect = 1.0 + 0.2 * np.sin(2 * np.pi * date.month / 12.0)
            weekday_effect = 1.0 + 0.3 * (date.dayofweek >= 5)  # Weekend boost
            
            # Random noise
            noise = np.random.normal(1.0, 0.1)
            
            # Calculate demand
            demand = base_demand * month_effect * weekday_effect * noise
            demand = max(0, demand)
            
            # Create record
            record = {
                "sku": sku,
                "date": date,
                "time_idx": time_idx_map[date],
                "demand": demand,
                "month": date.month,
                "day_of_week": date.dayofweek,
                "day_of_month": date.day,
                "week_of_year": date.isocalendar()[1]
            }
            
            data.append(record)
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    # Add cyclical features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    return df

def train_model_directly(model, train_dataloader, val_dataloader, num_epochs=10):
    """Train the model directly using PyTorch"""
    print("Training model directly with PyTorch...")
    
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        
        for x, y_tuple in train_dataloader:
            # Move data to device
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            
            # Handle y correctly - it's a tuple
            if isinstance(y_tuple, tuple):
                y = y_tuple[0].to(device)  # Extract the first element of the tuple
            else:
                y = y_tuple.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x)
            if isinstance(outputs, tuple):
                predictions = outputs[0]
            else:
                predictions = outputs
                
            # Calculate loss
            loss = model.loss(predictions, y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_train_loss += loss.item()
            train_batches += 1
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x, y_tuple in val_dataloader:
                # Move data to device
                x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                
                # Handle y correctly - it's a tuple
                if isinstance(y_tuple, tuple):
                    y = y_tuple[0].to(device)  # Extract the first element of the tuple
                else:
                    y = y_tuple.to(device)
                
                # Forward pass
                outputs = model(x)
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                # Calculate loss
                loss = model.loss(predictions, y)
                
                # Track loss
                epoch_val_loss += loss.item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.savefig("training_history.png")
    print("Training history saved as training_history.png")
    
    return model

def generate_forecasts(model, training, df):
    """Generate forecasts for all SKUs"""
    print("Generating forecasts...")
    
    # Get parameters
    max_prediction_length = training.max_prediction_length
    max_encoder_length = training.max_encoder_length
    
    # Move model to the right device
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Define training cutoff (matching what was used for training)
    training_cutoff = df["time_idx"].max() - max_prediction_length
    print(f"Training cutoff time_idx: {training_cutoff}")
    
    # For each SKU
    for sku in df["sku"].unique():
        try:
            # Get data for this SKU
            sku_data = df[df["sku"] == sku].copy()
            
            # Make sure we have enough data for encoder
            if len(sku_data) < max_encoder_length:
                print(f"Not enough data for {sku} to create a forecast (needs {max_encoder_length} points, has {len(sku_data)})")
                continue
                
            # Create a dataframe specifically for prediction
            # We need to use data up to the cutoff for prediction
            sku_data_sorted = sku_data.sort_values("time_idx")
            
            # Get the right window of data for prediction
            # This is crucial - we need to use the data at the cutoff point
            encoder_data = sku_data_sorted[sku_data_sorted["time_idx"] <= training_cutoff].tail(max_encoder_length).copy()
            
            # Ensure proper categorical types
            encoder_data["sku"] = encoder_data["sku"].astype(str)
            encoder_data["month"] = encoder_data["month"].astype(str)
            encoder_data["day_of_week"] = encoder_data["day_of_week"].astype(str)
            
            # Debug info
            print(f"Encoder data for {sku}:")
            print(f"  - Shape: {encoder_data.shape}")
            print(f"  - time_idx range: {encoder_data['time_idx'].min()} to {encoder_data['time_idx'].max()}")
            print(f"  - date range: {encoder_data['date'].min()} to {encoder_data['date'].max()}")
            
            # Create prediction dataset
            # We need to pass predict=True to create a prediction dataset
            encoder_data = encoder_data.reset_index(drop=True)
            pred_data = TimeSeriesDataSet.from_dataset(
                training, 
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
                    
                    # Get prediction values - handle potential shape issues
                    mean_prediction = predictions.mean(1).squeeze().cpu().numpy()
                    
                    # Handle single-value predictions (can happen with batch size 1)
                    if len(mean_prediction.shape) == 0:
                        mean_prediction = np.array([mean_prediction.item()])
                    
                    lower_bound = predictions.quantile(0.05, dim=1).squeeze().cpu().numpy()
                    if len(lower_bound.shape) == 0:
                        lower_bound = np.array([lower_bound.item()])
                        
                    upper_bound = predictions.quantile(0.95, dim=1).squeeze().cpu().numpy()
                    if len(upper_bound.shape) == 0:
                        upper_bound = np.array([upper_bound.item()])
                    
                    # Create future dates - start from the end of training cutoff
                    last_date = encoder_data["date"].max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=max_prediction_length)
                    
                    # Compare with actual data for validation
                    actual_future = sku_data_sorted[sku_data_sorted["time_idx"] > training_cutoff].head(max_prediction_length)
                    
                    # Plot
                    plt.figure(figsize=(12, 6))
                    # Historical data
                    plt.plot(sku_data_sorted["date"], sku_data_sorted["demand"], 'b-', label="Historical", alpha=0.5)
                    
                    # Mark the training cutoff
                    cutoff_date = sku_data_sorted[sku_data_sorted["time_idx"] == training_cutoff]["date"].iloc[0]
                    plt.axvline(x=cutoff_date, color='g', linestyle='--', label="Forecast Start")
                    
                    # Forecast
                    plt.plot(future_dates[:len(mean_prediction)], mean_prediction, "r-", label="Forecast")
                    plt.fill_between(
                        future_dates[:len(mean_prediction)],
                        lower_bound,
                        upper_bound,
                        color="red",
                        alpha=0.2,
                        label="90% Prediction Interval"
                    )
                    
                    # If we have actual data beyond cutoff, show it
                    if len(actual_future) > 0:
                        plt.plot(
                            actual_future["date"],
                            actual_future["demand"],
                            'g-', 
                            label="Actual Future Values", 
                            alpha=0.7
                        )
                    
                    plt.title(f"Demand Forecast for {sku}")
                    plt.xlabel("Date")
                    plt.ylabel("Demand")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{sku}_forecast.png")
                    print(f"Forecast for {sku} saved as {sku}_forecast.png")

        except Exception as e:
            print(f"Error generating forecast for {sku}: {e}")
    
    return True

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    df = create_synthetic_data()
    
    # Print data info
    print(f"Dataset shape: {df.shape}")
    print(f"Number of SKUs: {df['sku'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Convert types for modeling
    df["sku"] = df["sku"].astype(str)
    df["month"] = df["month"].astype(str)
    df["day_of_week"] = df["day_of_week"].astype(str)
    
    # Define parameters
    max_prediction_length = 30
    max_encoder_length = 60
    
    # Define training cutoff
    training_cutoff = df["time_idx"].max() - max_prediction_length
    
    # Create training dataset
    training_data = df[df["time_idx"] <= training_cutoff]
    
    # Create TimeSeriesDataset
    print("Creating dataset...")
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
    
    # Create data loaders
    batch_size = 64
    print("Creating data loaders...")
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = training.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
    )
    
    # Train model
    model = train_model_directly(model, train_dataloader, val_dataloader, num_epochs=5)
    
    # Generate forecasts
    generate_forecasts(model, training, df)
    
    print("Process completed successfully!")
    return df, model

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def create_synthetic_data(start_date="2022-01-01", end_date="2023-12-31", num_skus=5):
    """Generate synthetic demand data with seasonality and patterns"""
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Create SKUs
    skus = [f"SKU_{i}" for i in range(1, num_skus + 1)]
    
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

def load_real_data(file_path):
    """Load real demand data from a file"""
    df = pd.read_csv(file_path)
    
    # Ensure date is in datetime format
    df["date"] = pd.to_datetime(df["date"])
    
    # Create time index if not present
    if "time_idx" not in df.columns:
        dates = df["date"].sort_values().unique()
        time_idx_map = {date: idx for idx, date in enumerate(dates)}
        df["time_idx"] = df["date"].map(time_idx_map)
    
    # Add cyclical features if not present
    if "month_sin" not in df.columns:
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["week_of_year"] = df["date"].dt.isocalendar().week
        
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    return df

def prepare_data_for_training(df, max_prediction_length=35):
    """Prepare data for training by converting types and splitting"""
    # Convert types for modeling
    df["sku"] = df["sku"].astype(str)
    df["month"] = df["month"].astype(str)
    df["day_of_week"] = df["day_of_week"].astype(str)
    
    # Define training cutoff
    training_cutoff = df["time_idx"].max() - max_prediction_length
    
    # Create training dataset
    training_data = df[df["time_idx"] <= training_cutoff]
    validation_data = df[df["time_idx"] > training_cutoff]
    
    return training_data, validation_data, training_cutoff
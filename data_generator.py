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
    
    # Add price data for cross-effects analysis
    df['price'] = df['sku'].apply(lambda x: float(x.split('_')[1]) * 10).astype(float)
    # Add random variations
    df['price'] = df.apply(lambda row: row['price'] * (0.9 + 0.2 * np.random.random()), axis=1)
    # Add occasional promotions
    df['promo'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
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

def add_lag_features(df, lags=[7, 14, 28]):
    """Add lagged demand features"""
    print("Adding lag features...")
    for lag in lags:
        df[f'demand_lag_{lag}'] = df.groupby("sku")["demand"].shift(lag)
    return df

def add_rolling_statistics(df, windows=[7, 14, 28]):
    """Add rolling demand statistics"""
    print("Adding rolling statistics...")
    for window in windows:
        df[f'demand_rolling_mean_{window}'] = df.groupby("sku")["demand"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'demand_rolling_std_{window}'] = df.groupby("sku")["demand"].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
    return df

def add_holiday_indicators(df):
    """Add explicit holiday indicators"""
    print("Adding holiday indicators...")
    # This is a simplified approach - in practice, use a calendar library
    # Create US major holiday indicators (example)
    holidays = [
        # New Year's Day
        "2022-01-01", "2023-01-01",
        # Memorial Day (last Monday in May)
        "2022-05-30", "2023-05-29",
        # Independence Day
        "2022-07-04", "2023-07-04",
        # Labor Day (first Monday in September)
        "2022-09-05", "2023-09-04",
        # Thanksgiving (fourth Thursday in November)
        "2022-11-24", "2023-11-23",
        # Christmas
        "2022-12-25", "2023-12-25"
    ]
    
    # Convert to datetime
    holiday_dates = pd.to_datetime(holidays)
    
    # Create holiday flag
    df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
    
    # Add days until and days since closest holiday
    df['days_to_holiday'] = float('inf')
    df['days_since_holiday'] = float('inf')
    
    for holiday in holiday_dates:
        days_to = (holiday - df['date']).dt.days
        days_since = (df['date'] - holiday).dt.days
        
        # Only consider future holidays for days_to
        days_to = pd.Series([d if d > 0 else float('inf') for d in days_to])
        # Only consider past holidays for days_since
        days_since = pd.Series([d if d > 0 else float('inf') for d in days_since])
        
        df['days_to_holiday'] = df[['days_to_holiday']].join(
            days_to.rename('new_days_to')).min(axis=1)
        df['days_since_holiday'] = df[['days_since_holiday']].join(
            days_since.rename('new_days_since')).min(axis=1)
    
    # Replace infinite values
    df['days_to_holiday'] = df['days_to_holiday'].replace(float('inf'), 365)
    df['days_since_holiday'] = df['days_since_holiday'].replace(float('inf'), 365)
    
    return df

def handle_cross_effects_features(df):
    """Handle cross-effects features by filling or dropping NAs"""
    print("Handling cross-effects features...")
    
    # Identify cross-effect columns
    cross_effect_cols = [col for col in df.columns if 
                         col.startswith('avg_related_') or 
                         col.startswith('min_related_') or 
                         col.startswith('max_related_') or
                         col.startswith('num_related_') or 
                         col == 'joint_promos']
    
    # For each cross-effect column, fill NAs or drop the column if too many NAs
    for col in cross_effect_cols:
        na_pct = df[col].isna().mean()
        
        if na_pct > 0.8:  # If more than 80% are NA, drop the column
            print(f"Dropping column {col} (has {na_pct:.1%} NA values)")
            df = df.drop(columns=[col])
        else:
            # For price ratio columns, fill with 1.0 (neutral price ratio)
            if 'price_ratio' in col:
                print(f"Filling NA values in {col} with 1.0")
                df[col] = df[col].fillna(1.0)
            # For count columns, fill with 0
            else:
                print(f"Filling NA values in {col} with 0")
                df[col] = df[col].fillna(0)
    
    return df

def prepare_data_for_training(df, max_prediction_length=35):
    """Prepare data for training by converting types and splitting"""
    # Add enhanced features
    df = add_lag_features(df)
    df = add_rolling_statistics(df)
    df = add_holiday_indicators(df)
    
    # Handle cross-effects features if present
    df = handle_cross_effects_features(df)
    
    # Convert types for modeling
    df["sku"] = df["sku"].astype(str)
    df["month"] = df["month"].astype(str)
    df["day_of_week"] = df["day_of_week"].astype(str)
    df["is_holiday"] = df["is_holiday"].astype(str)
    
    # Define training cutoff
    training_cutoff = df["time_idx"].max() - max_prediction_length
    
    # Create training dataset
    training_data = df[df["time_idx"] <= training_cutoff]
    validation_data = df[df["time_idx"] > training_cutoff]
    
    # Fill NAs in lag features
    for col in df.columns:
        if col.startswith('demand_lag_') or col.startswith('demand_rolling_'):
            training_data[col] = training_data[col].fillna(0)
            validation_data[col] = validation_data[col].fillna(0)
    
    # Final NA check - ensure all numeric columns are filled
    numeric_cols = training_data.select_dtypes(include=['float', 'int']).columns
    for col in numeric_cols:
        training_data[col] = training_data[col].fillna(0)
        validation_data[col] = validation_data[col].fillna(0)
        
    print("Training data shape after preparation:", training_data.shape)
    print("Validation data shape after preparation:", validation_data.shape)
    
    # List columns with any remaining NA values as a double-check
    cols_with_na = [col for col in training_data.columns if training_data[col].isna().any()]
    if cols_with_na:
        print("Warning: These columns still have NA values:", cols_with_na)
        # Fill remaining NAs as a last resort
        for col in cols_with_na:
            if training_data[col].dtype in [np.float64, np.int64]:
                training_data[col] = training_data[col].fillna(0)
                validation_data[col] = validation_data[col].fillna(0)
            else:
                # For categorical columns, convert NA to string "NA"
                training_data[col] = training_data[col].fillna("NA")
                validation_data[col] = validation_data[col].fillna("NA")
    
    return training_data, validation_data, training_cutoff
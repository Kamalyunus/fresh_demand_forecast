import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_synthetic_data(start_date="2020-01-01", end_date="2023-12-31", num_skus=100):
    """
    Generate synthetic demand data with seasonality and patterns
    Returns data in format required by main.py
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Create SKUs with different patterns
    skus = [f"SKU_{i}" for i in range(1, num_skus + 1)]
    
    # Create dataframe
    data = []
    
    # Generate data for each SKU
    for sku in skus:
        # Base demand varies by SKU
        base_demand = np.random.randint(50, 500)  # Increased range for more variation
        base_price = float(sku.split('_')[1]) * 10  # Base price for each SKU
        
        # Determine segment (affects seasonality)
        segment = np.random.choice(['year_round', 'highly_seasonal', 'semi_seasonal', 'new_sku'], 
                                 p=[0.3, 0.3, 0.3, 0.1])  # Adjusted probabilities
        
        for date in dates:
            # Base seasonality
            month_effect = 1.0 + 0.3 * np.sin(2 * np.pi * date.month / 12.0)  # Increased seasonality
            
            # Weekend effect
            weekday_effect = 1.0 + 0.4 * (date.dayofweek >= 5)  # Increased weekend boost
            
            # Segment-specific patterns
            if segment == 'highly_seasonal':
                seasonality = 1.0 + 0.7 * np.sin(2 * np.pi * date.month / 12.0)  # Increased seasonality
            elif segment == 'semi_seasonal':
                seasonality = 1.0 + 0.4 * np.sin(2 * np.pi * date.month / 12.0)  # Increased seasonality
            elif segment == 'new_sku':
                # New SKUs have increasing trend
                days_since_start = (date - pd.to_datetime(start_date)).days
                trend = 1.0 + 0.002 * days_since_start  # Increased trend
                seasonality = 1.0
            else:  # year_round
                seasonality = 1.0
            
            # Random noise
            noise = np.random.normal(1.0, 0.15)  # Increased noise
            
            # Calculate volume
            volume = base_demand * month_effect * weekday_effect * seasonality * noise
            if segment == 'new_sku':
                volume = volume * trend
            volume = max(0, int(volume))
            
            # Create record
            record = {
                "sku": sku,
                "date": date,
                "volume": volume,
                "segment": segment,
                "base_price": base_price,
                "month": date.month,
                "day_of_week": date.dayofweek,
                "day_of_month": date.day,
                "week_of_year": date.isocalendar()[1],
                "holiday": 0,  # Default to no holiday
                "promotion": 0,  # Default to no promotion
                "special_event": 0,  # Default to no special event
                "is_holiday": 0,  # Binary holiday indicator
                "is_promotion": 0,  # Binary promotion indicator
                "promotion_depth": 0.0,  # Default promotion depth
                "days_until_holiday": 365,  # Default days until holiday
                "days_since_holiday": 365,  # Default days since holiday
            }
            
            data.append(record)
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    # Add cyclical features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    # Add price data
    df['price'] = df['base_price'] * df.apply(lambda row: 0.9 + 0.2 * np.random.random(), axis=1)
    
    # Add holiday indicators
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
    holiday_dates = pd.to_datetime(holidays)
    df.loc[df['date'].isin(holiday_dates), ['holiday', 'is_holiday']] = 1
    
    # Calculate days until and since holidays
    for date in holiday_dates:
        days_until = (date - df['date']).dt.days
        days_since = (df['date'] - date).dt.days
        
        # Update days until next holiday
        mask = (days_until > 0) & (days_until < df['days_until_holiday'])
        df.loc[mask, 'days_until_holiday'] = days_until[mask]
        
        # Update days since last holiday
        mask = (days_since > 0) & (days_since < df['days_since_holiday'])
        df.loc[mask, 'days_since_holiday'] = days_since[mask]
    
    # Add promotion indicators and depth (10% chance of promotion)
    promotion_mask = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    df.loc[promotion_mask == 1, ['promotion', 'is_promotion']] = 1
    df.loc[promotion_mask == 1, 'promotion_depth'] = np.random.uniform(0.1, 0.5, size=promotion_mask.sum())
    
    # Add special event indicators (5% chance of special event)
    df['special_event'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
    
    # Calculate average historical volume for each SKU
    df['avg_historical_volume'] = df.groupby('sku')['volume'].transform('mean')
    
    # Save to CSV
    df.to_csv('data.csv', index=False)
    print(f"Generated sample data with {len(df)} records and {df['sku'].nunique()} SKUs")
    print("Data saved to data.csv")
    
    return df

def load_real_data(file_path):
    """Load real demand data from a file"""
    df = pd.read_csv(file_path)
    
    # Ensure date is in datetime format
    df["date"] = pd.to_datetime(df["date"])
    
    # Add time-based features if not present
    if "month" not in df.columns:
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Add cyclical features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    return df

def add_lag_features(df, lags=[7, 14, 28]):
    """Add lagged volume features"""
    print("Adding lag features...")
    for lag in lags:
        df[f'volume_lag_{lag}'] = df.groupby("sku")["volume"].shift(lag)
    return df

def add_rolling_statistics(df, windows=[7, 14, 28]):
    """Add rolling volume statistics"""
    print("Adding rolling statistics...")
    for window in windows:
        df[f'volume_rolling_mean_{window}'] = df.groupby("sku")["volume"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'volume_rolling_std_{window}'] = df.groupby("sku")["volume"].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
    return df

def add_holiday_indicators(df):
    """Add holiday indicators"""
    print("Adding holiday indicators...")
    # Create US major holiday indicators
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
    
    holiday_dates = pd.to_datetime(holidays)
    df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
    
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
    training_cutoff = df["date"].max() - pd.Timedelta(days=max_prediction_length)
    
    # Create training dataset
    training_data = df[df["date"] <= training_cutoff]
    validation_data = df[df["date"] > training_cutoff]
    
    # Fill NAs in lag features
    for col in df.columns:
        if col.startswith('volume_lag_') or col.startswith('volume_rolling_'):
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
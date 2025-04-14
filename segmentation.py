# segmentation.py
"""
SKU segmentation module for demand forecasting

This module implements logic to segment SKUs based on their demand patterns:
- Year-round Items: Sales in ≥90% of days; stable but with some variation
- Highly Seasonal Items: Zero sales for ≥60% of year; clear season boundaries
- Semi-seasonal Items: Present year-round but with high variation (>100% coefficient)
- New SKUs: Less than 60 days of historical data
"""

import pandas as pd
import numpy as np
from datetime import timedelta

def segment_skus(df):
    """
    Segment SKUs based on their demand patterns
    
    Args:
        df: DataFrame with 'sku', 'date', and 'demand' columns
        
    Returns:
        DataFrame with 'sku' and 'segment' columns
    """
    # Initialize results
    results = []
    
    # Get unique SKUs
    skus = df['sku'].unique()
    
    for sku in skus:
        # Get data for this SKU
        sku_data = df[df['sku'] == sku].copy()
        sku_data = sku_data.sort_values('date')
        
        # Check if it's a new SKU (less than 60 days of data)
        total_days = (sku_data['date'].max() - sku_data['date'].min()).days + 1
        if total_days < 60:
            segment = "new_sku"
        else:
            # Calculate key metrics
            # 1. Percentage of days with sales
            total_possible_days = (df['date'].max() - df['date'].min()).days + 1
            
            # Create a date range covering all possible days
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
            all_dates_df = pd.DataFrame({'date': date_range})
            
            # Merge with actual data to identify days with and without sales
            merged_df = all_dates_df.merge(sku_data[['date', 'demand']], on='date', how='left')
            merged_df['demand'] = merged_df['demand'].fillna(0)
            
            days_with_sales = (merged_df['demand'] > 0).sum()
            pct_days_with_sales = days_with_sales / len(merged_df) * 100
            
            # 2. Check seasonality - percentage of year with zero sales
            zero_sales_pct = (merged_df['demand'] == 0).sum() / len(merged_df) * 100
            
            # 3. Calculate coefficient of variation (for non-zero demand)
            non_zero_demand = merged_df[merged_df['demand'] > 0]['demand']
            if len(non_zero_demand) > 0:
                cv = (non_zero_demand.std() / non_zero_demand.mean()) * 100
            else:
                cv = 0
            
            # Determine segment based on rules
            if pct_days_with_sales >= 90:
                # Year-round items
                segment = "year_round"
            elif zero_sales_pct >= 60:
                # Highly seasonal items
                segment = "seasonal"
            elif cv > 100:
                # Semi-seasonal items (high variation)
                segment = "semi_seasonal"
            else:
                # Default to year-round if no other condition is met
                segment = "year_round"
        
        # Add to results
        results.append({
            'sku': sku,
            'segment': segment
        })
    
    # Convert to DataFrame
    segments_df = pd.DataFrame(results)
    
    return segments_df

def identify_season_boundaries(df, seasonal_skus, min_season_length=14, gap_threshold=7):
    """
    Identify season boundaries for seasonal SKUs
    
    Args:
        df: DataFrame with 'sku', 'date', and 'demand' columns
        seasonal_skus: List of SKUs identified as seasonal
        min_season_length: Minimum length of a season in days
        gap_threshold: Number of consecutive zero-demand days to consider as a season boundary
        
    Returns:
        Dictionary mapping SKUs to lists of season periods (start_date, end_date)
    """
    season_boundaries = {}
    
    for sku in seasonal_skus:
        # Get data for this SKU
        sku_data = df[df['sku'] == sku].copy()
        sku_data = sku_data.sort_values('date')
        
        # Create a complete date range
        date_range = pd.date_range(start=sku_data['date'].min(), end=sku_data['date'].max(), freq='D')
        full_date_df = pd.DataFrame({'date': date_range})
        
        # Merge with actual data
        merged_df = full_date_df.merge(sku_data[['date', 'demand']], on='date', how='left')
        merged_df['demand'] = merged_df['demand'].fillna(0)
        
        # Identify periods of consecutive non-zero demand
        merged_df['has_demand'] = merged_df['demand'] > 0
        merged_df['streak_id'] = (merged_df['has_demand'] != merged_df['has_demand'].shift()).cumsum()
        
        # Get groups of consecutive days with demand
        seasons = []
        for streak_id, group in merged_df[merged_df['has_demand']].groupby('streak_id'):
            if len(group) >= min_season_length:
                seasons.append((group['date'].min(), group['date'].max()))
        
        # Merge adjacent seasons if gap is smaller than threshold
        merged_seasons = []
        if seasons:
            current_season = seasons[0]
            
            for i in range(1, len(seasons)):
                current_end = current_season[1]
                next_start = seasons[i][0]
                
                # Calculate gap
                gap = (next_start - current_end).days - 1
                
                if gap <= gap_threshold:
                    # Merge the seasons
                    current_season = (current_season[0], seasons[i][1])
                else:
                    # Add current season and start a new one
                    merged_seasons.append(current_season)
                    current_season = seasons[i]
            
            # Add the last season
            merged_seasons.append(current_season)
        
        season_boundaries[sku] = merged_seasons
    
    return season_boundaries

def find_related_products(df, top_n=3, correlation_threshold=0.5):
    """
    Find related products based on demand correlations
    
    Args:
        df: DataFrame with 'sku', 'date', and 'demand' columns
        top_n: Number of top related products to return for each SKU
        correlation_threshold: Minimum correlation to consider products as related
        
    Returns:
        Dictionary mapping SKUs to lists of related SKUs
    """
    # Create a pivot table with skus as columns and dates as rows
    pivot_df = df.pivot_table(index='date', columns='sku', values='demand', fill_value=0)
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Find related products for each SKU
    related_products = {}
    
    for sku in corr_matrix.columns:
        # Get correlations and sort by value (descending)
        correlations = corr_matrix[sku].sort_values(ascending=False)
        
        # Remove self-correlation
        correlations = correlations[correlations.index != sku]
        
        # Filter by threshold
        correlations = correlations[correlations >= correlation_threshold]
        
        # Take top N
        top_related = correlations.head(top_n).index.tolist()
        
        related_products[sku] = top_related
    
    return related_products

def segment_sku_in_production(sku_data, reference_date=None, days_threshold=60):
    """
    Segment a single SKU for production use
    
    Args:
        sku_data: DataFrame with 'date' and 'demand' columns for a single SKU
        reference_date: Date to use as reference for calculating history length
                        (defaults to max date in the data)
        days_threshold: Threshold for new SKUs
        
    Returns:
        String indicating the segment
    """
    # Sort by date
    sku_data = sku_data.sort_values('date')
    
    # Set reference date if not provided
    if reference_date is None:
        reference_date = sku_data['date'].max()
    
    # Check if it's a new SKU (less than threshold days of data)
    total_days = (sku_data['date'].max() - sku_data['date'].min()).days + 1
    if total_days < days_threshold:
        return "new_sku"
    
    # Calculate key metrics
    # 1. Percentage of days with sales
    # Create a date range covering all possible days
    date_range = pd.date_range(start=sku_data['date'].min(), end=sku_data['date'].max(), freq='D')
    all_dates_df = pd.DataFrame({'date': date_range})
    
    # Merge with actual data to identify days with and without sales
    merged_df = all_dates_df.merge(sku_data[['date', 'demand']], on='date', how='left')
    merged_df['demand'] = merged_df['demand'].fillna(0)
    
    days_with_sales = (merged_df['demand'] > 0).sum()
    pct_days_with_sales = days_with_sales / len(merged_df) * 100
    
    # 2. Check seasonality - percentage of year with zero sales
    zero_sales_pct = (merged_df['demand'] == 0).sum() / len(merged_df) * 100
    
    # 3. Calculate coefficient of variation (for non-zero demand)
    non_zero_demand = merged_df[merged_df['demand'] > 0]['demand']
    if len(non_zero_demand) > 0:
        cv = (non_zero_demand.std() / non_zero_demand.mean()) * 100
    else:
        cv = 0
    
    # Determine segment based on rules
    if pct_days_with_sales >= 90:
        # Year-round items
        return "year_round"
    elif zero_sales_pct >= 60:
        # Highly seasonal items
        return "seasonal"
    elif cv > 100:
        # Semi-seasonal items (high variation)
        return "semi_seasonal"
    else:
        # Default to year-round if no other condition is met
        return "year_round"

def get_sku_segments_report(df):
    """
    Generate a report of SKU segments with summary statistics
    
    Args:
        df: DataFrame with 'sku', 'date', and 'demand' columns
        
    Returns:
        DataFrame with segment statistics
    """
    # Segment SKUs
    segments_df = segment_skus(df)
    
    # Count SKUs in each segment
    segment_counts = segments_df['segment'].value_counts().reset_index()
    segment_counts.columns = ['segment', 'count']
    
    # Calculate percentage
    total_skus = len(segments_df)
    segment_counts['percentage'] = segment_counts['count'] / total_skus * 100
    
    # Get demand statistics by segment
    segment_stats = []
    
    for segment, group in segments_df.groupby('segment'):
        skus_in_segment = group['sku'].tolist()
        segment_data = df[df['sku'].isin(skus_in_segment)]
        
        # Calculate average demand
        avg_demand = segment_data['demand'].mean()
        
        # Calculate average days with sales
        avg_days_with_sales = 0
        avg_cv = 0
        
        for sku in skus_in_segment:
            sku_data = df[df['sku'] == sku]
            date_range = pd.date_range(start=sku_data['date'].min(), end=sku_data['date'].max(), freq='D')
            all_dates_df = pd.DataFrame({'date': date_range})
            merged_df = all_dates_df.merge(sku_data[['date', 'demand']], on='date', how='left')
            merged_df['demand'] = merged_df['demand'].fillna(0)
            
            days_with_sales = (merged_df['demand'] > 0).sum() / len(merged_df) * 100
            avg_days_with_sales += days_with_sales
            
            # Calculate CV
            non_zero_demand = merged_df[merged_df['demand'] > 0]['demand']
            if len(non_zero_demand) > 0:
                cv = (non_zero_demand.std() / non_zero_demand.mean()) * 100
                avg_cv += cv
        
        if len(skus_in_segment) > 0:
            avg_days_with_sales /= len(skus_in_segment)
            avg_cv /= len(skus_in_segment)
        
        segment_stats.append({
            'segment': segment,
            'avg_demand': avg_demand,
            'avg_days_with_sales_pct': avg_days_with_sales,
            'avg_cv': avg_cv
        })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(segment_stats)
    
    # Merge with counts
    report_df = segment_counts.merge(stats_df, on='segment')
    
    return report_df
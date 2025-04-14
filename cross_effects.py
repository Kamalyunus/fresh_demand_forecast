# cross_effects.py
"""
Modified version with better handling of missing values
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_price_elasticity(df, price_col='price', demand_col='demand', 
                               sku_col='sku', date_col='date', min_periods=30):
    """
    Calculate own-price elasticity for each SKU
    
    Args:
        df: DataFrame with price and demand data
        price_col: Name of the price column
        demand_col: Name of the demand column
        sku_col: Name of the SKU column
        date_col: Name of the date column
        min_periods: Minimum number of periods required to calculate elasticity
        
    Returns:
        DataFrame with SKU and own-price elasticity
    """
    logger.info("Calculating own-price elasticity")
    
    # Check if price column exists
    if price_col not in df.columns:
        logger.warning(f"Price column '{price_col}' not found in dataframe")
        return pd.DataFrame(columns=['sku', 'elasticity', 'elasticity_stdev', 'p_value', 'valid'])
    
    results = []
    
    for sku in df[sku_col].unique():
        # Get data for this SKU
        sku_data = df[df[sku_col] == sku].copy()
        
        # Skip if not enough data or no price variation
        if len(sku_data) < min_periods or sku_data[price_col].std() == 0:
            results.append({
                'sku': sku,
                'elasticity': np.nan,
                'elasticity_stdev': np.nan,
                'p_value': np.nan,
                'valid': False
            })
            continue
        
        try:
            # Take log of price and demand
            sku_data['log_price'] = np.log(sku_data[price_col].clip(lower=0.1))
            sku_data['log_demand'] = np.log(sku_data[demand_col].clip(lower=0.1))  # Avoid log(0)
            
            # Remove NaN values
            sku_data = sku_data.dropna(subset=['log_price', 'log_demand'])
            
            if len(sku_data) < min_periods:
                results.append({
                    'sku': sku,
                    'elasticity': np.nan,
                    'elasticity_stdev': np.nan,
                    'p_value': np.nan,
                    'valid': False
                })
                continue
            
            # Add constant for regression
            X = sm.add_constant(sku_data['log_price'])
            
            # Run regression
            model = OLS(sku_data['log_demand'], X).fit()
            
            # Get elasticity (the coefficient on log_price)
            elasticity = model.params['log_price']
            
            # Get standard error
            elasticity_stdev = model.bse['log_price']
            
            # Get p-value
            p_value = model.pvalues['log_price']
            
            results.append({
                'sku': sku,
                'elasticity': elasticity,
                'elasticity_stdev': elasticity_stdev,
                'p_value': p_value,
                'valid': True
            })
        except Exception as e:
            logger.warning(f"Error calculating elasticity for {sku}: {e}")
            results.append({
                'sku': sku,
                'elasticity': np.nan,
                'elasticity_stdev': np.nan,
                'p_value': np.nan,
                'valid': False
            })
    
    return pd.DataFrame(results)

def calculate_cross_price_elasticity(df, target_sku, candidate_skus=None, 
                                    price_col='price', demand_col='demand', 
                                    sku_col='sku', date_col='date', 
                                    min_periods=30, significance_level=0.1):
    """
    Calculate cross-price elasticity between a target SKU and candidate SKUs
    
    Args:
        df: DataFrame with price and demand data
        target_sku: SKU for which to calculate cross-price elasticity
        candidate_skus: List of SKUs to check for cross-effects (if None, use all)
        price_col: Name of the price column
        demand_col: Name of the demand column
        sku_col: Name of the SKU column
        date_col: Name of the date column
        min_periods: Minimum periods of overlap required
        significance_level: P-value threshold for significant elasticity
        
    Returns:
        DataFrame with cross-price elasticities
    """
    logger.info(f"Calculating cross-price elasticity for {target_sku}")
    
    # Check if price column exists
    if price_col not in df.columns:
        logger.warning(f"Price column '{price_col}' not found in dataframe")
        return pd.DataFrame()
    
    # Get data for target SKU
    target_data = df[df[sku_col] == target_sku].copy()
    
    if len(target_data) < min_periods:
        logger.warning(f"Not enough data for {target_sku}")
        return pd.DataFrame()
    
    # Set date as index for easier merging
    target_data = target_data.set_index(date_col)
    
    # Define candidate SKUs if not provided
    if candidate_skus is None:
        candidate_skus = [sku for sku in df[sku_col].unique() if sku != target_sku]
    
    results = []
    
    for candidate_sku in candidate_skus:
        # Skip self
        if candidate_sku == target_sku:
            continue
        
        # Get data for candidate SKU
        candidate_data = df[df[sku_col] == candidate_sku].copy()
        
        if len(candidate_data) < min_periods:
            continue
        
        # Set date as index
        candidate_data = candidate_data.set_index(date_col)
        
        # Rename columns to avoid confusion
        candidate_data = candidate_data.rename(columns={
            price_col: f'candidate_price',
            demand_col: f'candidate_demand'
        })
        
        # Merge data
        merged_data = target_data.join(candidate_data[[f'candidate_price', f'candidate_demand']], how='inner')
        
        # Skip if not enough overlapping periods
        if len(merged_data) < min_periods:
            continue
        
        try:
            # Take log of price and demand
            merged_data['log_target_demand'] = np.log(merged_data[demand_col].clip(lower=0.1))
            merged_data['log_candidate_price'] = np.log(merged_data['candidate_price'].clip(lower=0.1))
            merged_data['log_target_price'] = np.log(merged_data[price_col].clip(lower=0.1))
            
            # Remove NaN values
            merged_data = merged_data.dropna(subset=['log_target_demand', 'log_candidate_price', 'log_target_price'])
            
            if len(merged_data) < min_periods:
                continue
            
            # Add constant for regression
            X = sm.add_constant(merged_data[['log_target_price', 'log_candidate_price']])
            
            # Run regression
            model = OLS(merged_data['log_target_demand'], X).fit()
            
            # Get cross-price elasticity (coefficient on log_candidate_price)
            cross_elasticity = model.params['log_candidate_price']
            
            # Get standard error and p-value
            cross_elasticity_stdev = model.bse['log_candidate_price']
            p_value = model.pvalues['log_candidate_price']
            
            # Only include significant results
            if p_value <= significance_level:
                results.append({
                    'target_sku': target_sku,
                    'related_sku': candidate_sku,
                    'cross_elasticity': cross_elasticity,
                    'cross_elasticity_stdev': cross_elasticity_stdev,
                    'p_value': p_value,
                    'periods': len(merged_data),
                    'effect_type': 'substitute' if cross_elasticity < 0 else 'complement'
                })
        except Exception as e:
            logger.warning(f"Error calculating cross-elasticity between {target_sku} and {candidate_sku}: {e}")
    
    return pd.DataFrame(results)

def find_top_related_products(df, price_col='price', demand_col='demand', 
                             sku_col='sku', date_col='date', top_n=3, 
                             min_periods=30, significance_level=0.1):
    """
    Find top related products for each SKU based on cross-price elasticity
    
    Args:
        df: DataFrame with price and demand data
        price_col: Name of the price column
        demand_col: Name of the demand column
        sku_col: Name of the SKU column
        date_col: Name of the date column
        top_n: Number of top related products to return
        min_periods: Minimum periods of overlap required
        significance_level: P-value threshold for significant elasticity
        
    Returns:
        Dictionary mapping SKUs to lists of related SKUs
    """
    logger.info("Finding top related products")
    
    # Ensure we have a price column
    if price_col not in df.columns:
        logger.warning(f"Price column '{price_col}' not found in dataframe")
        # Return empty dictionary with all SKUs
        return {sku: [] for sku in df[sku_col].unique()}
    
    # Get unique SKUs
    skus = df[sku_col].unique()
    
    # Dictionary to store results
    related_products = {}
    
    for sku in skus:
        # Calculate cross-price elasticity
        cross_elasticity_df = calculate_cross_price_elasticity(
            df, sku, candidate_skus=None,
            price_col=price_col, demand_col=demand_col,
            sku_col=sku_col, date_col=date_col,
            min_periods=min_periods, significance_level=significance_level
        )
        
        if len(cross_elasticity_df) == 0:
            related_products[sku] = []
            continue
        
        # Sort by absolute elasticity strength
        cross_elasticity_df['abs_elasticity'] = cross_elasticity_df['cross_elasticity'].abs()
        sorted_df = cross_elasticity_df.sort_values('abs_elasticity', ascending=False)
        
        # Get top N related products
        top_related = sorted_df.head(top_n)
        
        related_products[sku] = [
            {
                'related_sku': row['related_sku'],
                'elasticity': row['cross_elasticity'],
                'effect_type': row['effect_type']
            }
            for _, row in top_related.iterrows()
        ]
    
    return related_products

def generate_cross_product_features(df, related_products_dict, price_col='price',
                                  sku_col='sku', date_col='date', promo_col=None):
    """
    Generate features that capture cross-product effects
    
    Args:
        df: DataFrame with price and demand data
        related_products_dict: Dictionary mapping SKUs to lists of related SKUs
        price_col: Name of the price column
        sku_col: Name of the SKU column
        date_col: Name of the date column
        promo_col: Name of the promotion column (if available)
        
    Returns:
        DataFrame with added cross-product features
    """
    logger.info("Generating cross-product features")
    
    # Create a copy of the dataframe
    enhanced_df = df.copy()
    
    # Check if we have price data - if not, return the original dataframe
    if price_col not in enhanced_df.columns:
        logger.warning(f"Price column '{price_col}' not found in dataframe")
        return enhanced_df
    
    # Initialize cross-product feature columns with zeros
    enhanced_df['avg_related_price_ratio'] = 0.0
    enhanced_df['min_related_price_ratio'] = 0.0 
    enhanced_df['max_related_price_ratio'] = 0.0
    
    if promo_col is not None and promo_col in df.columns:
        enhanced_df['num_related_promos'] = 0
        enhanced_df['joint_promos'] = 0
    
    # Create a pivot table of prices
    price_pivot = df.pivot_table(index=date_col, columns=sku_col, values=price_col, fill_value=0)
    
    # Create promotion pivot if promo column is available
    if promo_col is not None and promo_col in df.columns:
        promo_pivot = df.pivot_table(index=date_col, columns=sku_col, values=promo_col, fill_value=0)
    
    # Process each unique SKU
    for sku in enhanced_df[sku_col].unique():
        # Skip if SKU has no related products
        if sku not in related_products_dict or len(related_products_dict[sku]) == 0:
            continue
        
        # Get related products
        related_products = related_products_dict[sku]
        
        # Get indices for this SKU
        sku_indices = enhanced_df[enhanced_df[sku_col] == sku].index
        
        for idx in sku_indices:
            # Get date for this record
            date = enhanced_df.loc[idx, date_col]
            
            # Initialize features
            price_ratios = []
            related_promos = []
            joint_promos = 0
            
            for related_product in related_products:
                related_sku = related_product['related_sku']
                
                # Calculate price ratio if both prices are available
                if date in price_pivot.index and sku in price_pivot.columns and related_sku in price_pivot.columns:
                    own_price = price_pivot.loc[date, sku]
                    related_price = price_pivot.loc[date, related_sku]
                    
                    if own_price > 0 and related_price > 0:
                        price_ratio = related_price / own_price
                        price_ratios.append(price_ratio)
                
                # Check for promotions
                if promo_col is not None and promo_col in df.columns and date in promo_pivot.index and related_sku in promo_pivot.columns:
                    related_promo = promo_pivot.loc[date, related_sku]
                    related_promos.append(related_promo)
                    
                    # Check for joint promotion
                    if enhanced_df.loc[idx, promo_col] > 0 and related_promo > 0:
                        joint_promos += 1
            
            # Add features to the dataframe
            if price_ratios:
                enhanced_df.loc[idx, 'avg_related_price_ratio'] = np.mean(price_ratios)
                enhanced_df.loc[idx, 'min_related_price_ratio'] = np.min(price_ratios)
                enhanced_df.loc[idx, 'max_related_price_ratio'] = np.max(price_ratios)
            else:
                # Default to 1.0 (neutral price ratio) if no data
                enhanced_df.loc[idx, 'avg_related_price_ratio'] = 1.0
                enhanced_df.loc[idx, 'min_related_price_ratio'] = 1.0
                enhanced_df.loc[idx, 'max_related_price_ratio'] = 1.0
            
            if promo_col is not None and promo_col in df.columns and related_promos:
                enhanced_df.loc[idx, 'num_related_promos'] = sum(related_promos)
                enhanced_df.loc[idx, 'joint_promos'] = joint_promos
    
    # Fill any remaining NAs - sometimes pivot operations can create NAs
    price_ratio_cols = ['avg_related_price_ratio', 'min_related_price_ratio', 'max_related_price_ratio']
    for col in price_ratio_cols:
        enhanced_df[col] = enhanced_df[col].fillna(1.0)  # Neutral value
    
    if promo_col is not None and promo_col in df.columns:
        enhanced_df['num_related_promos'] = enhanced_df['num_related_promos'].fillna(0)
        enhanced_df['joint_promos'] = enhanced_df['joint_promos'].fillna(0)
    
    # Check for any remaining NAs in our added columns
    for col in enhanced_df.columns:
        if col not in df.columns and enhanced_df[col].isna().any():
            logger.warning(f"Column {col} still has {enhanced_df[col].isna().sum()} NA values after processing")
    
    return enhanced_df
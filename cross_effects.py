# cross_effects.py
"""
Module for analyzing and incorporating cross-product effects in demand forecasting.

This module focuses on:
1. Detecting cannibalization (negative cross-price elasticity)
2. Detecting halo effects (positive cross-price elasticity)
3. Generating features that capture these relationships
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
    
    results = []
    
    for sku in tqdm(df[sku_col].unique()):
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
            sku_data['log_price'] = np.log(sku_data[price_col])
            sku_data['log_demand'] = np.log(sku_data[demand_col].clip(lower=0.1))  # Avoid log(0)
            
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
    
    for candidate_sku in tqdm(candidate_skus):
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
            merged_data['log_candidate_price'] = np.log(merged_data['candidate_price'])
            merged_data['log_target_price'] = np.log(merged_data[price_col])
            
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
        raise ValueError(f"Price column '{price_col}' not found in dataframe")
    
    # Get unique SKUs
    skus = df[sku_col].unique()
    
    # Dictionary to store results
    related_products = {}
    
    for sku in tqdm(skus):
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
    
    # Create a pivot table of prices
    price_pivot = df.pivot_table(index=date_col, columns=sku_col, values=price_col)
    
    # Create promotion pivot if promo column is available
    if promo_col is not None and promo_col in df.columns:
        promo_pivot = df.pivot_table(index=date_col, columns=sku_col, values=promo_col, fill_value=0)
    
    # Process each row
    for index, row in tqdm(enhanced_df.iterrows(), total=len(enhanced_df)):
        sku = row[sku_col]
        date = row[date_col]
        
        # Skip if SKU has no related products
        if sku not in related_products_dict or len(related_products_dict[sku]) == 0:
            continue
        
        # Get related products
        related_products = related_products_dict[sku]
        
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
            if promo_col is not None and date in promo_pivot.index and related_sku in promo_pivot.columns:
                related_promo = promo_pivot.loc[date, related_sku]
                related_promos.append(related_promo)
                
                # Check for joint promotion
                if row[promo_col] > 0 and related_promo > 0:
                    joint_promos += 1
        
        # Add features to the dataframe
        if price_ratios:
            enhanced_df.loc[index, 'avg_related_price_ratio'] = np.mean(price_ratios)
            enhanced_df.loc[index, 'min_related_price_ratio'] = np.min(price_ratios)
            enhanced_df.loc[index, 'max_related_price_ratio'] = np.max(price_ratios)
        
        if promo_col is not None and related_promos:
            enhanced_df.loc[index, 'num_related_promos'] = sum(related_promos)
            enhanced_df.loc[index, 'joint_promos'] = joint_promos
    
    return enhanced_df

def add_category_aggregates(df, sku_col='sku', date_col='date', demand_col='demand', 
                          category_col=None, num_periods=7):
    """
    Add category-level aggregate features
    
    Args:
        df: DataFrame with demand data
        sku_col: Name of the SKU column
        date_col: Name of the date column
        demand_col: Name of the demand column
        category_col: Name of the category column (if available)
        num_periods: Number of periods for trend calculation
        
    Returns:
        DataFrame with added category aggregate features
    """
    logger.info("Adding category aggregate features")
    
    # Create a copy of the dataframe
    enhanced_df = df.copy()
    
    # Determine categories
    if category_col is not None and category_col in df.columns:
        # Use provided category column
        categories = df[category_col].unique()
        category_mapping = df[[sku_col, category_col]].drop_duplicates().set_index(sku_col)[category_col].to_dict()
    else:
        # Treat all SKUs as one category
        categories = ['all']
        category_mapping = {sku: 'all' for sku in df[sku_col].unique()}
    
    # Create pivot of demand
    demand_pivot = df.pivot_table(index=date_col, columns=sku_col, values=demand_col, fill_value=0)
    
    # Calculate category aggregates for each date
    category_totals = {}
    category_trends = {}
    
    for category in categories:
        # Get SKUs in this category
        category_skus = [sku for sku, cat in category_mapping.items() if cat == category]
        
        # Skip if no SKUs in this category
        if not category_skus:
            continue
        
        # Calculate daily category total
        category_demand = demand_pivot[category_skus].sum(axis=1)
        category_totals[category] = category_demand
        
        # Calculate trend
        category_trends[category] = category_demand.rolling(window=num_periods).mean()
    
    # Add features to each row
    for index, row in tqdm(enhanced_df.iterrows(), total=len(enhanced_df)):
        sku = row[sku_col]
        date = row[date_col]
        category = category_mapping.get(sku)
        
        if category is None or date not in demand_pivot.index:
            continue
        
        # Category total demand
        if category in category_totals and date in category_totals[category].index:
            category_total = category_totals[category].loc[date]
            enhanced_df.loc[index, 'category_total_demand'] = category_total
            
            # Calculate SKU's share of category
            if category_total > 0:
                enhanced_df.loc[index, 'category_share'] = row[demand_col] / category_total
        
        # Category trend
        if category in category_trends and date in category_trends[category].index:
            category_trend = category_trends[category].loc[date]
            
            if pd.notnull(category_trend):
                enhanced_df.loc[index, 'category_trend'] = category_trend
                
                # Calculate trend growth rate
                prev_date_idx = demand_pivot.index.get_loc(date) - num_periods
                if prev_date_idx >= 0:
                    prev_date = demand_pivot.index[prev_date_idx]
                    if prev_date in category_trends[category].index:
                        prev_trend = category_trends[category].loc[prev_date]
                        if pd.notnull(prev_trend) and prev_trend > 0:
                            enhanced_df.loc[index, 'category_growth_rate'] = category_trend / prev_trend - 1
    
    return enhanced_df

def main(df, price_col='price', demand_col='demand', sku_col='sku', 
        date_col='date', promo_col=None, category_col=None, 
        top_n=3, min_periods=30):
    """
    Main function to analyze and incorporate cross-product effects
    
    Args:
        df: DataFrame with price, demand, and optional promo data
        price_col: Name of the price column
        demand_col: Name of the demand column
        sku_col: Name of the SKU column
        date_col: Name of the date column
        promo_col: Name of the promotion column (if available)
        category_col: Name of the category column (if available)
        top_n: Number of top related products to return
        min_periods: Minimum periods of overlap required
        
    Returns:
        Enhanced DataFrame with cross-product features
    """
    # 1. Find top related products
    related_products = find_top_related_products(
        df,
        price_col=price_col,
        demand_col=demand_col,
        sku_col=sku_col,
        date_col=date_col,
        top_n=top_n,
        min_periods=min_periods
    )
    
    # 2. Generate cross-product features
    enhanced_df = generate_cross_product_features(
        df,
        related_products,
        price_col=price_col,
        sku_col=sku_col,
        date_col=date_col,
        promo_col=promo_col
    )
    
    # 3. Add category aggregates
    enhanced_df = add_category_aggregates(
        enhanced_df,
        sku_col=sku_col,
        date_col=date_col,
        demand_col=demand_col,
        category_col=category_col
    )
    
    return enhanced_df, related_products

if __name__ == "__main__":
    # This is a sample usage - in reality, you would load your own data
    import numpy as np
    
    # Create synthetic data
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    skus = [f"SKU_{i}" for i in range(1, 6)]
    categories = ["Fresh", "Dairy", "Fresh", "Dairy", "Fresh"]
    
    data = []
    
    for sku_idx, sku in enumerate(skus):
        base_price = 5 + sku_idx
        base_demand = 100 - sku_idx * 10
        category = categories[sku_idx]
        
        for date in dates:
            # Price with some randomness and occasional promotions
            is_promo = np.random.random() < 0.1
            price = base_price * (0.8 if is_promo else 1.0) * np.random.normal(1, 0.05)
            
            # Demand with price elasticity, seasonality, and noise
            price_effect = -0.5 * (price / base_price - 1)
            season_effect = 0.2 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(0, 0.1)
            
            demand = base_demand * (1 + price_effect + season_effect + noise)
            demand = max(0, demand)
            
            data.append({
                "sku": sku,
                "date": date,
                "price": price,
                "demand": demand,
                "promo": 1 if is_promo else 0,
                "category": category
            })
    
    sample_df = pd.DataFrame(data)
    
    # Run the main function
    enhanced_df, related_products = main(
        sample_df,
        price_col='price',
        demand_col='demand',
        sku_col='sku',
        date_col='date',
        promo_col='promo',
        category_col='category'
    )
    
    # Print results
    print("Enhanced DataFrame columns:", enhanced_df.columns.tolist())
    print("\nSample of enhanced data:")
    print(enhanced_df.sample(5))
    
    print("\nRelated products:")
    for sku, related in related_products.items():
        if related:
            print(f"{sku}:")
            for product in related:
                print(f"  - {product['related_sku']} ({product['effect_type']}, elasticity: {product['elasticity']:.3f})")
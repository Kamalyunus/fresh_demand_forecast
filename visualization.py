import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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
    """Plot feature importance from TFT model with better error handling"""
    import os
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Skip feature importance plotting completely
        # Create a simple fallback plot
        plt.figure(figsize=(10, 5))
        
        # Just extract a few parameter names and their magnitudes as a simple approximation
        importance_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) <= 2:  # Only use simple parameters
                if 'weight' in name or 'embedding' in name:
                    importance_dict[name] = param.abs().mean().item()
        
        # Sort by importance and take top 20
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create a simple horizontal bar chart
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, labels)
        plt.xlabel('Parameter Magnitude (approximation of importance)')
        plt.title('Model Parameter Magnitudes (Simple Feature Importance Approximation)')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{output_dir}/feature_importance_simple.png")
        print(f"Simple parameter magnitude plot saved to {output_dir}/feature_importance_simple.png")
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

def plot_evaluation_metrics(results_df, output_dir="./plots"):
    """Plot evaluation metrics by SKU"""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(results_df) == 0:
        print("No evaluation results to plot")
        return
    
    # Plot MAE, RMSE by SKU
    plt.figure(figsize=(12, 6))
    x = range(len(results_df))
    width = 0.35
    
    plt.bar(x, results_df["mae"], width, label="MAE")
    plt.bar([i + width for i in x], results_df["rmse"], width, label="RMSE")
    
    plt.xlabel("SKU")
    plt.ylabel("Error")
    plt.title("MAE and RMSE by SKU")
    plt.xticks([i + width/2 for i in x], results_df["sku"], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_metrics_by_sku.png")
    plt.close()
    
    # Plot MAPE by SKU
    plt.figure(figsize=(12, 6))
    plt.bar(results_df["sku"], results_df["mape"], color="orange")
    plt.xlabel("SKU")
    plt.ylabel("MAPE (%)")
    plt.title("MAPE by SKU")
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mape_by_sku.png")
    plt.close()
    
    # Plot Bias by SKU
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df["sku"], results_df["bias_pct"], color="skyblue")
    
    # Color bars based on bias direction (positive = underforecast, negative = overforecast)
    for i, bias in enumerate(results_df["bias_pct"]):
        if bias < 0:
            bars[i].set_color("salmon")  # Reddish for overforecast
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel("SKU")
    plt.ylabel("Bias (%)")
    plt.title("Forecast Bias by SKU (Positive = Underforecast, Negative = Overforecast)")
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bias_by_sku.png")
    plt.close()
    
    # Plot Over/Under forecast percentages
    plt.figure(figsize=(12, 6))
    x = range(len(results_df))
    width = 0.35
    
    plt.bar(x, results_df["over_forecast_pct"], width, label="Over Forecast %", color="salmon")
    plt.bar([i + width for i in x], results_df["under_forecast_pct"], width, label="Under Forecast %", color="skyblue")
    
    plt.xlabel("SKU")
    plt.ylabel("Percentage (%)")
    plt.title("Over and Under Forecast Percentages by SKU")
    plt.xticks([i + width/2 for i in x], results_df["sku"], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/over_under_forecast_by_sku.png")
    plt.close()
    
    # Create a correlation heatmap of metrics
    metrics_cols = ["mae", "rmse", "mape", "bias", "bias_pct", "over_forecast_pct", "under_forecast_pct"]
    corr_matrix = results_df[metrics_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Correlation Between Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_correlation.png")
    plt.close()
    
    print(f"Evaluation metric plots saved to {output_dir}/")

def plot_forecast_vs_actual(forecast_df, results_df, output_dir="./plots"):
    """Create scatter plots of forecast vs actual values with error analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dataframe with forecast and actual values
    forecast_actual = forecast_df[["sku", "date", "forecast", "actual"]].dropna()
    
    if len(forecast_actual) == 0:
        print("No forecast vs actual data to plot")
        return
    
    # Global scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(forecast_actual["actual"], forecast_actual["forecast"], alpha=0.5)
    
    # Add reference line (perfect forecast)
    max_val = max(forecast_actual["actual"].max(), forecast_actual["forecast"].max())
    min_val = min(forecast_actual["actual"].min(), forecast_actual["forecast"].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("Actual Demand")
    plt.ylabel("Forecasted Demand")
    plt.title("Forecast vs Actual Values")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast_vs_actual.png")
    plt.close()
    
    # Scatter plot by SKU (for top 6 SKUs by volume)
    top_skus = forecast_actual.groupby("sku")["actual"].sum().nlargest(6).index.tolist()
    
    plt.figure(figsize=(15, 10))
    for i, sku in enumerate(top_skus):
        plt.subplot(2, 3, i+1)
        
        sku_data = forecast_actual[forecast_actual["sku"] == sku]
        plt.scatter(sku_data["actual"], sku_data["forecast"], alpha=0.6)
        
        # Add reference line
        max_val = max(sku_data["actual"].max(), sku_data["forecast"].max())
        min_val = min(sku_data["actual"].min(), sku_data["forecast"].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Actual")
        plt.ylabel("Forecast")
        plt.title(f"SKU: {sku}")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast_vs_actual_by_sku.png")
    plt.close()
    
    # Residuals analysis
    forecast_actual["residual"] = forecast_actual["actual"] - forecast_actual["forecast"]
    forecast_actual["percent_error"] = forecast_actual["residual"] / forecast_actual["actual"].clip(lower=1) * 100
    
    # Residuals histogram
    plt.figure(figsize=(12, 6))
    plt.hist(forecast_actual["residual"], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel("Residual (Actual - Forecast)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Forecast Residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_histogram.png")
    plt.close()
    
    # Residuals vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(forecast_actual["actual"], forecast_actual["residual"], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Actual Demand")
    plt.ylabel("Residual (Actual - Forecast)")
    plt.title("Residuals vs Actual Values")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_vs_actual.png")
    plt.close()
    
    print(f"Forecast vs actual analysis plots saved to {output_dir}/")

def plot_bias_analysis(results_df, output_dir="./plots"):
    """Create detailed bias analysis plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(results_df) == 0:
        print("No results data for bias analysis")
        return
    
    # Calculate percentages of SKUs with over/under forecasting tendency
    bias_direction = []
    for _, row in results_df.iterrows():
        if row['bias'] > 0:
            bias_direction.append('Underforecast')
        elif row['bias'] < 0:
            bias_direction.append('Overforecast')
        else:
            bias_direction.append('Neutral')
    
    results_df['bias_direction'] = bias_direction
    
    direction_counts = results_df['bias_direction'].value_counts()
    
    # Pie chart of bias direction
    plt.figure(figsize=(10, 7))
    plt.pie(direction_counts, labels=direction_counts.index, autopct='%1.1f%%', 
            shadow=True, startangle=90, colors=['skyblue', 'salmon', 'lightgrey'])
    plt.axis('equal')
    plt.title('Forecast Bias Direction Distribution')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bias_direction_pie.png")
    plt.close()
    
    # Create bias severity categories
    def bias_category(bias_pct):
        if abs(bias_pct) < 5:
            return "Low bias (< 5%)"
        elif abs(bias_pct) < 15:
            return "Moderate bias (5-15%)"
        elif abs(bias_pct) < 30:
            return "High bias (15-30%)"
        else:
            return "Severe bias (> 30%)"
    
    results_df['bias_category'] = results_df['bias_pct'].apply(bias_category)
    category_counts = results_df['bias_category'].value_counts()
    
    # Bar chart of bias categories
    plt.figure(figsize=(12, 6))
    bars = plt.bar(category_counts.index, category_counts.values, color='skyblue')
    plt.xlabel('Bias Category')
    plt.ylabel('Number of SKUs')
    plt.title('Distribution of Forecast Bias Severity')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bias_severity_distribution.png")
    plt.close()
    
    # Relationship between bias and MAPE
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['bias_pct'].abs(), results_df['mape'], alpha=0.7)
    plt.xlabel('Absolute Bias Percentage')
    plt.ylabel('MAPE')
    plt.title('Relationship Between Bias and MAPE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bias_vs_mape.png")
    plt.close()
    
    print(f"Bias analysis plots saved to {output_dir}/")

def plot_prediction_intervals(forecast_df, output_dir="./plots", sku_sample=5):
    """
    Plot prediction interval widths and coverage analysis
    
    Args:
        forecast_df: DataFrame with forecast data including percentiles
        output_dir: Directory to save plots
        sku_sample: Number of SKUs to sample for individual plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if forecast_df has percentile columns
    has_percentiles = all(f'p{p}' in forecast_df.columns for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    
    if not has_percentiles:
        print("Forecast data does not contain percentile columns")
        return
    
    # Calculate prediction interval widths
    if 'pi_60' not in forecast_df.columns:
        forecast_df["pi_60"] = forecast_df["p80"] - forecast_df["p20"]  # 60% prediction interval
        forecast_df["pi_70"] = forecast_df["p85"] - forecast_df["p15"]  # 70% prediction interval
        forecast_df["pi_80"] = forecast_df["p90"] - forecast_df["p10"]  # 80% prediction interval
        forecast_df["pi_90"] = forecast_df["p95"] - forecast_df["p5"]   # 90% prediction interval
    
    # Create a plot of average prediction interval width by SKU
    pi_widths = forecast_df.groupby("sku")[["pi_60", "pi_70", "pi_80", "pi_90"]].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    x = range(len(pi_widths))
    width = 0.2
    
    plt.bar([i - 1.5*width for i in x], pi_widths["pi_60"], width, label="60% PI", color="lightblue")
    plt.bar([i - 0.5*width for i in x], pi_widths["pi_70"], width, label="70% PI", color="skyblue")
    plt.bar([i + 0.5*width for i in x], pi_widths["pi_80"], width, label="80% PI", color="royalblue")
    plt.bar([i + 1.5*width for i in x], pi_widths["pi_90"], width, label="90% PI", color="darkblue")
    
    plt.xlabel("SKU")
    plt.ylabel("Prediction Interval Width")
    plt.title("Average Prediction Interval Width by SKU")
    plt.xticks(x, pi_widths["sku"], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_interval_widths.png")
    plt.close()
    
    # Calculate prediction interval relative widths (as % of forecast value)
    forecast_df_with_forecast = forecast_df[forecast_df["forecast"] > 0].copy()
    forecast_df_with_forecast["pi_60_rel"] = forecast_df_with_forecast["pi_60"] / forecast_df_with_forecast["forecast"] * 100
    forecast_df_with_forecast["pi_70_rel"] = forecast_df_with_forecast["pi_70"] / forecast_df_with_forecast["forecast"] * 100
    forecast_df_with_forecast["pi_80_rel"] = forecast_df_with_forecast["pi_80"] / forecast_df_with_forecast["forecast"] * 100
    forecast_df_with_forecast["pi_90_rel"] = forecast_df_with_forecast["pi_90"] / forecast_df_with_forecast["forecast"] * 100
    
    # Create a plot of average relative prediction interval width by SKU
    pi_rel_widths = forecast_df_with_forecast.groupby("sku")[["pi_60_rel", "pi_70_rel", "pi_80_rel", "pi_90_rel"]].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    x = range(len(pi_rel_widths))
    width = 0.2
    
    plt.bar([i - 1.5*width for i in x], pi_rel_widths["pi_60_rel"], width, label="60% PI", color="lightblue")
    plt.bar([i - 0.5*width for i in x], pi_rel_widths["pi_70_rel"], width, label="70% PI", color="skyblue")
    plt.bar([i + 0.5*width for i in x], pi_rel_widths["pi_80_rel"], width, label="80% PI", color="royalblue")
    plt.bar([i + 1.5*width for i in x], pi_rel_widths["pi_90_rel"], width, label="90% PI", color="darkblue")
    
    plt.xlabel("SKU")
    plt.ylabel("Relative Prediction Interval Width (%)")
    plt.title("Average Relative Prediction Interval Width by SKU")
    plt.xticks(x, pi_rel_widths["sku"], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/relative_prediction_interval_widths.png")
    plt.close()
    
    # Calculate prediction interval coverage (% of actual values falling within interval)
    # Only for points where we have actual values
    forecast_with_actual = forecast_df.dropna(subset=["actual"]).copy()
    
    if len(forecast_with_actual) > 0:
        # Calculate coverage
        forecast_with_actual["in_pi_60"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p20"]) & 
                                         (forecast_with_actual["actual"] <= forecast_with_actual["p80"]))
        forecast_with_actual["in_pi_70"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p15"]) & 
                                         (forecast_with_actual["actual"] <= forecast_with_actual["p85"]))
        forecast_with_actual["in_pi_80"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p10"]) & 
                                         (forecast_with_actual["actual"] <= forecast_with_actual["p90"]))
        forecast_with_actual["in_pi_90"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p5"]) & 
                                         (forecast_with_actual["actual"] <= forecast_with_actual["p95"]))
        
        # Calculate coverage percentages by SKU
        coverage = forecast_with_actual.groupby("sku")[["in_pi_60", "in_pi_70", "in_pi_80", "in_pi_90"]].mean() * 100
        coverage = coverage.reset_index()
        
        # Plot coverage
        plt.figure(figsize=(12, 6))
        x = range(len(coverage))
        width = 0.2
        
        plt.bar([i - 1.5*width for i in x], coverage["in_pi_60"], width, label="60% PI", color="lightblue")
        plt.bar([i - 0.5*width for i in x], coverage["in_pi_70"], width, label="70% PI", color="skyblue")
        plt.bar([i + 0.5*width for i in x], coverage["in_pi_80"], width, label="80% PI", color="royalblue")
        plt.bar([i + 1.5*width for i in x], coverage["in_pi_90"], width, label="90% PI", color="darkblue")
        
        # Add reference lines for expected coverage
        plt.axhline(y=60, color='lightblue', linestyle='--', alpha=0.5)
        plt.axhline(y=70, color='skyblue', linestyle='--', alpha=0.5)
        plt.axhline(y=80, color='royalblue', linestyle='--', alpha=0.5)
        plt.axhline(y=90, color='darkblue', linestyle='--', alpha=0.5)
        
        plt.xlabel("SKU")
        plt.ylabel("Coverage (%)")
        plt.title("Prediction Interval Coverage by SKU")
        plt.xticks(x, coverage["sku"], rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_interval_coverage.png")
        plt.close()
        
        # Plot overall coverage
        overall_coverage = forecast_with_actual[["in_pi_60", "in_pi_70", "in_pi_80", "in_pi_90"]].mean() * 100
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(["60% PI", "70% PI", "80% PI", "90% PI"], 
                       overall_coverage.values, 
                       color=["lightblue", "skyblue", "royalblue", "darkblue"])
        
        # Add reference lines for expected coverage
        plt.axhline(y=60, color='lightblue', linestyle='--', alpha=0.5, label="Expected 60%")
        plt.axhline(y=70, color='skyblue', linestyle='--', alpha=0.5, label="Expected 70%")
        plt.axhline(y=80, color='royalblue', linestyle='--', alpha=0.5, label="Expected 80%")
        plt.axhline(y=90, color='darkblue', linestyle='--', alpha=0.5, label="Expected 90%")
        
        plt.xlabel("Prediction Interval")
        plt.ylabel("Coverage (%)")
        plt.title("Overall Prediction Interval Coverage")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_prediction_interval_coverage.png")
        plt.close()
    
    # Sample a few SKUs for detailed prediction interval analysis
    if sku_sample > 0:
        sample_skus = np.random.choice(forecast_df["sku"].unique(), 
                                       size=min(sku_sample, len(forecast_df["sku"].unique())), 
                                       replace=False)
        
        for sku in sample_skus:
            sku_forecast = forecast_df[forecast_df["sku"] == sku].copy()
            
            # Plot the detailed prediction intervals
            plt.figure(figsize=(12, 6))
            
            # Plot prediction intervals with decreasing opacity
            plt.fill_between(
                sku_forecast["date"],
                sku_forecast["p5"],
                sku_forecast["p95"],
                color="blue",
                alpha=0.1,
                label="90% Interval"
            )
            
            plt.fill_between(
                sku_forecast["date"],
                sku_forecast["p10"],
                sku_forecast["p90"],
                color="blue",
                alpha=0.1,
                label="80% Interval"
            )
            
            plt.fill_between(
                sku_forecast["date"],
                sku_forecast["p20"],
                sku_forecast["p80"],
                color="blue",
                alpha=0.1,
                label="60% Interval"
            )
            
            # Plot median and mean
            plt.plot(sku_forecast["date"], sku_forecast["p50"], 'b-', 
                    label="Median (P50)", linewidth=1.5)
            plt.plot(sku_forecast["date"], sku_forecast["forecast"], 'r-', 
                    label="Mean", linewidth=1.5)
            
            # Plot actual values if available
            has_actuals = ~sku_forecast["actual"].isna().all()
            if has_actuals:
                plt.plot(
                    sku_forecast["date"],
                    sku_forecast["actual"],
                    'g-', 
                    label="Actual", 
                    alpha=0.7,
                    linewidth=2
                )
            
            plt.title(f"Prediction Intervals for {sku}")
            plt.xlabel("Date")
            plt.ylabel("Demand")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f"{output_dir}/{sku}_prediction_intervals.png")
            plt.close()
    
    print(f"Prediction interval analysis plots saved to {output_dir}/")

def generate_evaluation_report(forecast_df, results_df, avg_metrics, output_dir="./plots"):
    """Generate comprehensive evaluation report with all metrics and visualizations"""
    # Create the basic evaluation metric plots
    plot_evaluation_metrics(results_df, output_dir)
    
    # Create forecast vs actual analysis
    plot_forecast_vs_actual(forecast_df, results_df, output_dir)
    
    # Create bias analysis
    plot_bias_analysis(results_df, output_dir)
    
    # Plot prediction interval analysis if we have percentiles
    has_percentiles = all(f'p{p}' in forecast_df.columns for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    if has_percentiles:
        plot_prediction_intervals(forecast_df, output_dir)
    
    # Generate textual report
    report_path = f"{output_dir}/evaluation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("==========================================\n")
        f.write("DEMAND FORECASTING EVALUATION REPORT\n")
        f.write("==========================================\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write("------------------------------------------\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n\nDETAILED METRICS BY SKU\n")
        f.write("------------------------------------------\n")
        f.write(results_df.to_string(index=False))
        
        f.write("\n\nBIAS ANALYSIS\n")
        f.write("------------------------------------------\n")
        bias_direction_counts = results_df['bias_direction'].value_counts()
        for direction, count in bias_direction_counts.items():
            f.write(f"{direction}: {count} SKUs ({count/len(results_df)*100:.1f}%)\n")
        
        # Calculate average metrics by bias direction
        f.write("\nAverage Metrics by Bias Direction:\n")
        bias_group_metrics = results_df.groupby('bias_direction')[['mae', 'rmse', 'mape']].mean()
        f.write(bias_group_metrics.to_string())
        
        # Prediction interval analysis if available
        if has_percentiles:
            forecast_with_actual = forecast_df.dropna(subset=["actual"]).copy()
            if len(forecast_with_actual) > 0:
                # Calculate coverage
                forecast_with_actual["in_pi_60"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p20"]) & 
                                                (forecast_with_actual["actual"] <= forecast_with_actual["p80"]))
                forecast_with_actual["in_pi_70"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p15"]) & 
                                                (forecast_with_actual["actual"] <= forecast_with_actual["p85"]))
                forecast_with_actual["in_pi_80"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p10"]) & 
                                                (forecast_with_actual["actual"] <= forecast_with_actual["p90"]))
                forecast_with_actual["in_pi_90"] = ((forecast_with_actual["actual"] >= forecast_with_actual["p5"]) & 
                                                (forecast_with_actual["actual"] <= forecast_with_actual["p95"]))
                
                overall_coverage = forecast_with_actual[["in_pi_60", "in_pi_70", "in_pi_80", "in_pi_90"]].mean() * 100
                
                f.write("\n\nPREDICTION INTERVAL COVERAGE\n")
                f.write("------------------------------------------\n")
                f.write(f"60% Prediction Interval: {overall_coverage['in_pi_60']:.1f}% (Expected: 60.0%)\n")
                f.write(f"70% Prediction Interval: {overall_coverage['in_pi_70']:.1f}% (Expected: 70.0%)\n")
                f.write(f"80% Prediction Interval: {overall_coverage['in_pi_80']:.1f}% (Expected: 80.0%)\n")
                f.write(f"90% Prediction Interval: {overall_coverage['in_pi_90']:.1f}% (Expected: 90.0%)\n")
                
                # Add calibration error
                calibration_error = np.mean([
                    abs(overall_coverage['in_pi_60'] - 60),
                    abs(overall_coverage['in_pi_70'] - 70),
                    abs(overall_coverage['in_pi_80'] - 80),
                    abs(overall_coverage['in_pi_90'] - 90)
                ])
                
                f.write(f"\nAverage Calibration Error: {calibration_error:.1f}%\n")
        
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("------------------------------------------\n")
        
        # Generate recommendations based on the metrics
        avg_bias = avg_metrics["Average Bias"]
        avg_bias_pct = avg_metrics["Average Bias %"]
        avg_mape = avg_metrics["Average MAPE"]
        under_pct = avg_metrics["Average Under Forecast %"]
        over_pct = avg_metrics["Average Over Forecast %"]
        
        if abs(avg_bias_pct) > 10:
            if avg_bias_pct > 0:
                f.write("• There is a significant tendency to UNDERFORECAST. Consider adjusting\n")
                f.write("  the model to increase its predictions by approximately {:.1f}%.\n".format(avg_bias_pct))
            else:
                f.write("• There is a significant tendency to OVERFORECAST. Consider adjusting\n")
                f.write("  the model to decrease its predictions by approximately {:.1f}%.\n".format(abs(avg_bias_pct)))
        
        if avg_mape > 30:
            f.write("• The forecasting accuracy is below expectations (MAPE: {:.1f}%).\n".format(avg_mape))
            f.write("  Consider additional feature engineering or model tuning.\n")
        
        if abs(under_pct - over_pct) > 20:
                    if under_pct > over_pct:
                        f.write("• The model has a strong bias toward underforecasting ({:.1f}% of points).\n".format(under_pct))
                        f.write("  Review loss function to potentially penalize underforecasting more.\n")
                    else:
                        f.write("• The model has a strong bias toward overforecasting ({:.1f}% of points).\n".format(over_pct))
                        f.write("  Review loss function to potentially penalize overforecasting more.\n")
                
                    # Prediction interval recommendations
                    if has_percentiles and len(forecast_with_actual) > 0:
                        if calibration_error > 10:
                            f.write("• The prediction intervals are not well-calibrated (Average error: {:.1f}%).\n".format(calibration_error))
                            if overall_coverage['in_pi_90'] < 85:
                                f.write("  The model is overconfident. Consider adjusting the uncertainty modeling.\n")
                            elif overall_coverage['in_pi_90'] > 95:
                                f.write("  The model is underconfident. Prediction intervals are too wide.\n")
                    
                    f.write("\nEvaluation report and visualizations saved to the plots directory.\n")
            
        print(f"Evaluation report saved to {report_path}")
        return report_path
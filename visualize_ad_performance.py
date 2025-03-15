#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import re
import sys

# On Windows, there is a problem OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add the directory containing digital_advertising.py to the Python path
# This allows importing without modifying the original file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import necessary components from digital_advertising.py
try:
    from digital_advertising import (
        ModelHandler, AdOptimizationEnv, generate_synthetic_data, feature_columns
    )
    print("Successfully imported components from digital_advertising.py")
except ImportError as e:
    print(f"Warning: Could not import from digital_advertising.py: {e}")
    print("Will use simplified functionality without direct model integration")
    
    # Define minimal versions of required components for visualization
    feature_columns = [
        "competitiveness", "difficulty_score", "organic_rank", "organic_clicks", 
        "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", 
        "ad_roas", "conversion_rate", "cost_per_click"
    ]
    
    def generate_synthetic_data(num_samples=1000):
        """Generate synthetic ad data for visualization if original function not available"""
        data = {
            "keyword": [f"Keyword_{i}" for i in range(num_samples)],
            "competitiveness": np.random.uniform(0, 1, num_samples),
            "difficulty_score": np.random.uniform(0, 1, num_samples),
            "organic_rank": np.random.randint(1, 11, num_samples),
            "organic_clicks": np.random.randint(50, 5000, num_samples),
            "organic_ctr": np.random.uniform(0.01, 0.3, num_samples),
            "paid_clicks": np.random.randint(10, 3000, num_samples),
            "paid_ctr": np.random.uniform(0.01, 0.25, num_samples),
            "ad_spend": np.random.uniform(10, 10000, num_samples),
            "ad_conversions": np.random.randint(0, 500, num_samples),
            "ad_roas": np.random.uniform(0.5, 5, num_samples),
            "conversion_rate": np.random.uniform(0.01, 0.3, num_samples),
            "cost_per_click": np.random.uniform(0.1, 10, num_samples),
            "cost_per_acquisition": np.random.uniform(5, 500, num_samples),
            "previous_recommendation": np.random.choice([0, 1], size=num_samples),
            "impression_share": np.random.uniform(0.1, 1.0, num_samples),
            "conversion_value": np.random.uniform(0, 10000, num_samples)
        }
        return pd.DataFrame(data)


def parse_tensorboard_logs(logdir):
    """
    Parse TensorBoard event files and extract metrics.
    
    Args:
        logdir (str): Path to TensorBoard log directory
        
    Returns:
        dict: Dictionary containing extracted metrics
    """
    print(f"Parsing TensorBoard logs from {logdir}")
    
    # Check if directory exists
    if not os.path.exists(logdir):
        print(f"Warning: TensorBoard log directory {logdir} does not exist")
        return {}
    
    metrics = defaultdict(list)
    
    # Find all event files in the directory
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"Warning: No TensorBoard event files found in {logdir}")
        print("Will proceed with dataset analysis only")
        return {}
    
    print(f"Found {len(event_files)} event files")
    
    # Process each event file
    for event_file in event_files:
        print(f"Processing {event_file}")
        
        try:
            # Use a more lenient reload approach with a timeout
            event_acc = EventAccumulator(event_file, size_guidance={
                'scalars': 0,  # 0 means load all
                'tensors': 0
            })
            try:
                event_acc.Reload()
            except Exception as e:
                print(f"Warning: Error during reload: {e}. Attempting to continue anyway.")
            
            # Get available tags (metrics)
            try:
                tags = event_acc.Tags()
                scalar_tags = tags.get('scalars', [])
                
                print(f"Available scalar metrics: {scalar_tags}")
                
                # Extract data for each scalar tag
                for tag in scalar_tags:
                    try:
                        events = event_acc.Scalars(tag)
                        
                        # For each event, extract step, wall_time, and value
                        for event in events:
                            metrics[tag].append({
                                'step': event.step,
                                'time': event.wall_time,
                                'value': event.value
                            })
                    except Exception as e:
                        print(f"Warning: Error extracting scalar {tag}: {e}")
                        
                # Also extract text data if available
                text_tags = tags.get('tensors', [])
                for tag in text_tags:
                    if 'Feature Columns' in tag or 'Num Keywords' in tag:
                        try:
                            events = event_acc.Tensors(tag)
                            for event in events:
                                # Extract text content from tensor event
                                metrics[tag].append({
                                    'step': event.step,
                                    'time': event.wall_time,
                                    'value': str(event.tensor_proto)
                                })
                        except Exception as e:
                            print(f"Warning: Error extracting text data for {tag}: {e}")
            except Exception as e:
                print(f"Warning: Error getting tags: {e}")
                
        except Exception as e:
            print(f"Warning: Error processing event file {event_file}: {e}")
            print("Continuing with next file.")
    
    # Convert lists to DataFrames for easier analysis
    metric_dfs = {}
    for tag, events in metrics.items():
        if events:
            try:
                metric_dfs[tag] = pd.DataFrame(events)
            except Exception as e:
                print(f"Warning: Could not convert {tag} to DataFrame: {e}")
            
    return metric_dfs


def visualize_training_metrics(metric_dfs, output_dir):
    """
    Create visualizations based on training metrics from TensorBoard.
    
    Args:
        metric_dfs (dict): Dictionary of DataFrames containing metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    if not metric_dfs:
        print("No metrics data available for visualization")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Check for key metrics
    loss_metric = next((m for m in metric_dfs.keys() if 'loss' in m.lower()), None)
    test_perf_metric = next((m for m in metric_dfs.keys() if 'test' in m.lower() and 'performance' in m.lower()), None)
    
    # 1. Plot training loss
    if loss_metric:
        loss_df = metric_dfs[loss_metric]
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_df['step'], loss_df['value'])
        plt.title('Training Loss Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)
        
        loss_plot_path = os.path.join(output_dir, "training_loss.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        saved_plots.append(loss_plot_path)
        print(f"Training loss plot saved to {loss_plot_path}")
        
    # 2. Plot test performance
    if test_perf_metric:
        test_df = metric_dfs[test_perf_metric]
        
        plt.figure(figsize=(10, 6))
        plt.plot(test_df['step'], test_df['value'])
        plt.title('Test Performance Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Test Reward')
        plt.grid(True, alpha=0.3)
        
        perf_plot_path = os.path.join(output_dir, "test_performance.png")
        plt.savefig(perf_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        saved_plots.append(perf_plot_path)
        print(f"Test performance plot saved to {perf_plot_path}")
        
    # 3. Combined metrics plot if both are available
    if loss_metric and test_perf_metric:
        loss_df = metric_dfs[loss_metric]
        test_df = metric_dfs[test_perf_metric]
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        color1 = 'tab:red'
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss Value', color=color1)
        ax1.plot(loss_df['step'], loss_df['value'], color=color1, alpha=0.7, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Test Reward', color=color2)
        ax2.plot(test_df['step'], test_df['value'], color=color2, alpha=0.7, label='Test Performance')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        plt.title('Training Progress')
        plt.grid(True, alpha=0.3)
        
        combined_plot_path = os.path.join(output_dir, "training_progress.png")
        plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        saved_plots.append(combined_plot_path)
        print(f"Combined training progress plot saved to {combined_plot_path}")
    
    return saved_plots


def visualize_keyword_performance(dataset, output_dir):
    """
    Create visualizations of keyword metrics in the dataset.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Avoid using any seaborn functions that might use problematic pandas options
    # Instead use only matplotlib directly
    
    # 1. ROAS vs Ad Spend relationship
    plt.figure(figsize=(12, 8))
    # Use matplotlib's scatter instead of seaborn
    plt.scatter(
        dataset["ad_spend"], 
        dataset["ad_roas"],
        c=dataset["competitiveness"],
        cmap="viridis",
        alpha=0.7,
        s=100
    )
    plt.colorbar(label="Competitiveness")
    plt.title("ROAS vs Ad Spend by Keyword Competitiveness")
    plt.xlabel("Ad Spend")
    plt.ylabel("Return on Ad Spend (ROAS)")
    plt.grid(True, alpha=0.3)
    
    # Add ROAS = 1 reference line (break-even point)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="Break-Even (ROAS = 1.0)")
    plt.legend()
    
    roas_plot_path = os.path.join(output_dir, "roas_vs_spend.png")
    plt.savefig(roas_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(roas_plot_path)
    
    # 2. CTR vs Competitiveness - Using standard scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(
        dataset["competitiveness"],
        dataset["paid_ctr"],
        c=dataset["organic_rank"],
        cmap="coolwarm",
        alpha=0.7,
        s=100
    )
    plt.colorbar(label="Organic Rank")
    plt.title("Click-Through Rate vs Keyword Competitiveness")
    plt.xlabel("Competitiveness")
    plt.ylabel("Paid CTR")
    plt.grid(True, alpha=0.3)
    
    ctr_plot_path = os.path.join(output_dir, "ctr_vs_competitiveness.png")
    plt.savefig(ctr_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(ctr_plot_path)
    
    # 3. Feature correlations heatmap - use matplotlib instead of seaborn
    plt.figure(figsize=(14, 12))
    # Calculate correlation matrix manually
    corr = dataset[feature_columns].corr().values
    
    # Create heatmap manually
    plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label="Correlation Coefficient")
    
    # Add feature names as ticks
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=90)
    plt.yticks(range(len(feature_columns)), feature_columns)
    
    # Add correlation values as text
    for i in range(len(feature_columns)):
        for j in range(len(feature_columns)):
            plt.text(j, i, f"{corr[i, j]:.2f}", 
                     ha="center", va="center", 
                     color="black" if abs(corr[i, j]) < 0.7 else "white")
    
    plt.title("Feature Correlation Matrix")
    
    corr_plot_path = os.path.join(output_dir, "feature_correlations.png")
    plt.savefig(corr_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(corr_plot_path)
    
    # 4. Key performance indicators distribution using simple histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Distribution of Key Performance Indicators", fontsize=16)
    
    # ROAS Distribution - simple histogram
    axes[0, 0].hist(dataset["ad_roas"], bins=30, color="green", alpha=0.7)
    axes[0, 0].axvline(x=1.0, color='r', linestyle='--', label="Break-even (ROAS=1)")
    axes[0, 0].set_title("ROAS Distribution")
    axes[0, 0].set_xlabel("Return on Ad Spend")
    axes[0, 0].legend()
    
    # CTR Distribution
    axes[0, 1].hist(dataset["paid_ctr"], bins=30, color="blue", alpha=0.7)
    axes[0, 1].set_title("Paid CTR Distribution")
    axes[0, 1].set_xlabel("Click-Through Rate")
    
    # Ad Spend Distribution
    axes[1, 0].hist(dataset["ad_spend"], bins=30, color="purple", alpha=0.7)
    axes[1, 0].set_title("Ad Spend Distribution")
    axes[1, 0].set_xlabel("Ad Spend")
    
    # Conversion Rate Distribution
    axes[1, 1].hist(dataset["conversion_rate"], bins=30, color="orange", alpha=0.7)
    axes[1, 1].set_title("Conversion Rate Distribution")
    axes[1, 1].set_xlabel("Conversion Rate")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    kpi_plot_path = os.path.join(output_dir, "kpi_distributions.png")
    plt.savefig(kpi_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(kpi_plot_path)
    
    return saved_plots


def visualize_investment_decision_strategies(dataset, output_dir):
    """
    Create visualizations that illustrate potential investment decision strategies.
    This doesn't require the trained model but provides insight into the decision space.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Create a synthetic decision column based on ROAS and CTR thresholds
    # This simulates what an agent might learn
    dataset = dataset.copy()
    dataset['decision'] = 'Don\'t Invest'
    
    # High ROAS strategy
    high_roas_mask = (dataset['ad_roas'] > 2.0)
    dataset.loc[high_roas_mask, 'decision'] = 'Invest (High ROAS)'
    
    # Balanced strategy for medium ROAS with good CTR
    balanced_mask = ((dataset['ad_roas'] > 1.2) & (dataset['ad_roas'] <= 2.0) & 
                     (dataset['paid_ctr'] > 0.15))
    dataset.loc[balanced_mask, 'decision'] = 'Invest (Balanced)'
    
    # Aggressive strategy for high CTR even with marginal ROAS
    aggressive_mask = ((dataset['ad_roas'] > 1.0) & (dataset['ad_roas'] <= 1.2) & 
                       (dataset['paid_ctr'] > 0.18))
    dataset.loc[aggressive_mask, 'decision'] = 'Invest (Aggressive)'
    
    # Create color map for decisions
    decision_colors = {
        "Invest (High ROAS)": "darkgreen", 
        "Invest (Balanced)": "limegreen", 
        "Invest (Aggressive)": "orange", 
        "Don't Invest": "red"
    }
    
    # Map decisions to colors for plotting
    point_colors = [decision_colors[decision] for decision in dataset['decision']]
    
    # 1. Decision Map: ROAS vs Ad Spend
    plt.figure(figsize=(14, 10))
    
    # Plot each decision category separately to create legend properly
    for decision, color in decision_colors.items():
        mask = dataset['decision'] == decision
        if mask.any():  # Only plot if at least one point exists
            plt.scatter(
                dataset.loc[mask, 'ad_spend'],
                dataset.loc[mask, 'ad_roas'],
                color=color,
                alpha=0.7,
                s=100,
                label=decision
            )
    
    plt.title("Investment Decision Strategy Map: ROAS vs Ad Spend")
    plt.xlabel("Ad Spend")
    plt.ylabel("Return on Ad Spend (ROAS)")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label="Break-Even (ROAS = 1.0)")
    plt.axhline(y=1.2, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(y=2.0, color='gray', linestyle='-.', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Decision Strategy", loc="upper right")
    
    decision_map_path = os.path.join(output_dir, "decision_strategy_map.png")
    plt.savefig(decision_map_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(decision_map_path)
    
    # 2. Decision Map: ROAS vs CTR
    plt.figure(figsize=(14, 10))
    
    # Plot each decision category separately
    for decision, color in decision_colors.items():
        mask = dataset['decision'] == decision
        if mask.any():  # Only plot if at least one point exists
            plt.scatter(
                dataset.loc[mask, 'paid_ctr'],
                dataset.loc[mask, 'ad_roas'],
                color=color,
                alpha=0.7,
                s=100,
                label=decision
            )
    
    plt.title("Investment Decision Strategy Map: ROAS vs CTR")
    plt.xlabel("Click-Through Rate (CTR)")
    plt.ylabel("Return on Ad Spend (ROAS)")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label="Break-Even (ROAS = 1.0)")
    plt.axhline(y=1.2, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(y=2.0, color='gray', linestyle='-.', alpha=0.7)
    plt.axvline(x=0.15, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=0.18, color='gray', linestyle='-.', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Decision Strategy", loc="upper right")
    
    decision_ctr_path = os.path.join(output_dir, "decision_strategy_ctr.png")
    plt.savefig(decision_ctr_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(decision_ctr_path)
    
    # 3. Strategy Distribution Analysis
    strategy_counts = dataset['decision'].value_counts()
    
    plt.figure(figsize=(12, 8))
    # Use matplotlib's bar instead of pandas plot
    plt.bar(
        strategy_counts.index,
        strategy_counts.values,
        color=[decision_colors[d] for d in strategy_counts.index]
    )
    plt.title("Distribution of Investment Decisions by Strategy")
    plt.xlabel("Investment Strategy")
    plt.ylabel("Number of Keywords")
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for i, count in enumerate(strategy_counts):
        plt.text(i, count + 5, str(count), ha='center')
    
    strategy_dist_path = os.path.join(output_dir, "strategy_distribution.png")
    plt.savefig(strategy_dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(strategy_dist_path)
    
    # 4. Expected Return Analysis by Strategy
    # Calculate potential returns for each keyword based on decision
    dataset['investment'] = dataset['ad_spend'] * (
        dataset['decision'].isin(['Invest (High ROAS)', 'Invest (Balanced)', 'Invest (Aggressive)'])
    ).astype(int)
    
    dataset['expected_return'] = dataset['investment'] * dataset['ad_roas']
    dataset['profit'] = dataset['expected_return'] - dataset['investment']
    
    # Aggregate by strategy - use manual aggregation instead of pandas groupby
    strategies = dataset['decision'].unique()
    roi_data = []
    
    for strategy in strategies:
        strategy_subset = dataset[dataset['decision'] == strategy]
        investment_sum = strategy_subset['investment'].sum()
        profit_sum = strategy_subset['profit'].sum()
        
        roi = 0
        if investment_sum > 0:
            roi = profit_sum / investment_sum
            
        roi_data.append({
            'strategy': strategy,
            'roi': roi,
            'color': decision_colors.get(strategy, 'gray')
        })
    
    # Plot ROI by strategy using matplotlib
    plt.figure(figsize=(12, 8))
    strategies = [item['strategy'] for item in roi_data]
    roi_values = [item['roi'] for item in roi_data]
    colors = [item['color'] for item in roi_data]
    
    bars = plt.bar(strategies, roi_values, color=colors)
    plt.title("Return on Investment by Strategy")
    plt.xlabel("Investment Strategy")
    plt.ylabel("ROI")
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01, 
            f'{height:.2f}', 
            ha='center', va='bottom'
        )
    
    roi_path = os.path.join(output_dir, "roi_by_strategy.png")
    plt.savefig(roi_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(roi_path)
    
    return saved_plots


def create_html_report(plots, output_dir, params=None):
    """
    Create an HTML report with all visualizations.
    
    Args:
        plots (dict): Dictionary with section names and plots
        output_dir (str): Output directory
        params (dict): Optional parameters extracted from logs
        
    Returns:
        str: Path to HTML report
    """
    report_path = os.path.join(output_dir, "ad_performance_report.html")
    
    # HTML template
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digital Advertising Performance Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #2980b9;
                margin-top: 30px;
            }
            .section {
                margin-bottom: 40px;
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .plot {
                text-align: center;
                margin: 20px 0;
            }
            .plot img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
            .caption {
                font-style: italic;
                color: #666;
                margin-top: 10px;
            }
            .params {
                background-color: #eee;
                padding: 15px;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
            }
            .timestamp {
                color: #777;
                font-size: 0.9em;
                text-align: right;
                margin-top: 30px;
            }
        </style>
    </head>
    <body>
        <h1>Digital Advertising Performance Analysis</h1>
    """
    
    # Add timestamp
    html += f"""
        <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    """
    
    # Add parameters section if available
    if params:
        html += """
        <div class="section">
            <h2>Training Parameters</h2>
            <div class="params">
        """
        
        for key, value in params.items():
            html += f"{key}: {value}\n"
        
        html += """
            </div>
        </div>
        """
    
    # Add each section with its plots
    for section_name, section_plots in plots.items():
        html += f"""
        <div class="section">
            <h2>{section_name}</h2>
        """
        
        for plot_path in section_plots:
            plot_filename = os.path.basename(plot_path)
            plot_rel_path = os.path.relpath(plot_path, output_dir)
            
            # Create caption from filename
            caption = plot_filename.replace('.png', '').replace('_', ' ').title()
            
            html += f"""
            <div class="plot">
                <img src="{plot_rel_path}" alt="{caption}">
                <div class="caption">{caption}</div>
            </div>
            """
        
        html += """
        </div>
        """
    
    # Close HTML document
    html += """
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated at {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Visualize Ad Performance from TensorBoard Logs")
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV (if None, generates synthetic data)")
    parser.add_argument("--output_dir", type=str, default="visualization_results", help="Output directory for visualizations")
    parser.add_argument("--model", type=str, default=None, help="Optional path to saved model for additional analysis")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for synthetic data generation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    # Parse TensorBoard logs with error handling
    try:
        metric_dfs = parse_tensorboard_logs(args.logdir)
    except Exception as e:
        print(f"Warning: Error parsing TensorBoard logs: {e}")
        print("Proceeding with dataset analysis only.")
        metric_dfs = {}
    
    # Extract parameters from logs if available
    extracted_params = {}
    if metric_dfs:
        for tag in metric_dfs:
            if isinstance(tag, str) and 'text' in tag.lower():
                try:
                    for _, row in metric_dfs[tag].iterrows():
                        # Attempt to extract parameter from text
                        text_value = str(row['value'])
                        param_match = re.search(r'([a-zA-Z_]+):\s*([\d\.]+)', text_value)
                        if param_match:
                            param_name, param_value = param_match.groups()
                            extracted_params[param_name] = param_value
                except Exception as e:
                    print(f"Warning: Error extracting parameters from {tag}: {e}")
    
    # Load or generate dataset with error handling
    try:
        if args.dataset and os.path.exists(args.dataset):
            print(f"Loading dataset from {args.dataset}")
            dataset = pd.read_csv(args.dataset)
        else:
            print(f"Generating synthetic dataset with {args.num_samples} samples")
            dataset = generate_synthetic_data(args.num_samples)
            dataset_path = os.path.join(args.output_dir, "synthetic_ad_data.csv")
            dataset.to_csv(dataset_path, index=False)
            print(f"Synthetic dataset saved to {dataset_path}")
    except Exception as e:
        print(f"Error with dataset: {e}")
        print("Generating emergency synthetic dataset")
        dataset = generate_synthetic_data(100)  # Smaller emergency dataset
    
    # Generate visualizations with error handling
    all_plots = {}
    
    # 1. Training metrics from TensorBoard
    if metric_dfs:
        try:
            all_plots["Training Progress"] = visualize_training_metrics(
                metric_dfs, 
                os.path.join(args.output_dir, "training_metrics")
            )
        except Exception as e:
            print(f"Warning: Error visualizing training metrics: {e}")
            all_plots["Training Progress"] = []
    
    # 2. Keyword performance metrics
    try:
        all_plots["Keyword Performance Analysis"] = visualize_keyword_performance(
            dataset,
            os.path.join(args.output_dir, "keyword_performance")
        )
    except Exception as e:
        print(f"Warning: Error visualizing keyword performance: {e}")
        all_plots["Keyword Performance Analysis"] = []
    
    # 3. Investment decision strategies
    try:
        all_plots["Investment Decision Strategies"] = visualize_investment_decision_strategies(
            dataset,
            os.path.join(args.output_dir, "decision_strategies")
        )
    except Exception as e:
        print(f"Warning: Error visualizing investment strategies: {e}")
        all_plots["Investment Decision Strategies"] = []
    
    # Create HTML report with all visualizations
    try:
        html_report_path = create_html_report(all_plots, args.output_dir, extracted_params)
        print(f"\nVisualization complete!")
        print(f"HTML report available at: {html_report_path}")
    except Exception as e:
        print(f"Warning: Error creating HTML report: {e}")
    
    print(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

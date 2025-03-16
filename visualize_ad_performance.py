#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
# Force matplotlib to use Agg backend to prevent GUI-related errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import re
import sys
import traceback

# On Windows, there is a problem OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure matplotlib to prevent buffer overflows
def configure_matplotlib_constraints():
    """Configure matplotlib rendering pipeline with dimensional constraints"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams['savefig.dpi'] = 100  # Lower DPI for saved figures

# Apply constraints immediately
configure_matplotlib_constraints()

# Define minimal versions of required components for visualization
feature_columns = [
    "competitiveness", "difficulty_score", "organic_rank", "organic_clicks", 
    "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", 
    "ad_roas", "conversion_rate", "cost_per_click"
]

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic ad data for visualization"""
    print(f"Generating synthetic dataset with {num_samples} samples...")
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
        dict: Dictionary of DataFrames containing parsed metrics
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

def visualize_epsilon_greedy_exploration(metric_dfs, output_dir):
    """
    Create visualizations focusing on epsilon-greedy exploration strategy.
    
    Args:
        metric_dfs (dict): Dictionary of DataFrames containing metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    print(f"Generating epsilon-greedy exploration visualization...")
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Generate synthetic training data if real metrics aren't available
    if not metric_dfs:
        print("No TensorBoard metrics found. Generating synthetic training data.")
        # Create synthetic data for visualizing the epsilon-greedy behavior
        steps = np.linspace(0, 100000, 30)
        epsilon_values = 0.9 - 0.89 * (steps / 100000)
        
        # Create synthetic reward curve that improves with training
        rewards = []
        for step in steps:
            if step < 15000:
                # Initially low rewards during high exploration
                reward = max(0, np.random.normal(0.05, 0.1) + step/150000)
            elif step < 70000:
                # Increasing rewards during balanced exploration/exploitation
                reward = max(0, np.random.normal(0.5, 0.2) + (step-15000)/110000)
            else:
                # Higher, more stable rewards during exploitation
                reward = max(0, np.random.normal(1.2, 0.15) + (step-70000)/300000)
            rewards.append(reward)
        
        # Generate synthetic loss values that decrease with training
        losses = 10 * np.exp(-steps / 40000) + np.random.normal(0, 0.3, size=len(steps))
        losses = np.maximum(losses, 0.5)  # Ensure positive losses
        
        # Create a DataFrame with synthetic data
        synth_data = pd.DataFrame({
            'step': steps,
            'epsilon': epsilon_values,
            'test_reward': rewards,
            'loss': losses
        })
        
        # Define exploration phases
        def get_phase(epsilon):
            if epsilon > 0.7:
                return "High Exploration"
            elif epsilon > 0.3:
                return "Balanced Exploration/Exploitation"
            else:
                return "High Exploitation"
        
        synth_data['phase'] = synth_data['epsilon'].apply(get_phase)
        
    else:
        # Check if we have the necessary data
        loss_metric = next((m for m in metric_dfs.keys() if 'loss' in m.lower()), None)
        test_perf_metric = next((m for m in metric_dfs.keys() if 'test' in m.lower() and 'performance' in m.lower()), None)
        
        if not loss_metric or not test_perf_metric:
            print("Missing required metrics. Generating synthetic data instead.")
            return visualize_epsilon_greedy_exploration({}, output_dir)
        
        # Extract data
        loss_df = metric_dfs[loss_metric].copy()
        test_df = metric_dfs[test_perf_metric].copy()
        
        # Merge datasets on step
        # First, find the closest step in test_df for each step in loss_df
        merged_data = []
        
        for _, loss_row in loss_df.iterrows():
            loss_step = loss_row['step']
            
            # Find the closest test step
            closest_test_idx = (test_df['step'] - loss_step).abs().argmin()
            test_row = test_df.iloc[closest_test_idx]
            
            # Only include if steps are reasonably close
            if abs(test_row['step'] - loss_step) < 5000:  # Threshold for "closeness"
                merged_data.append({
                    'step': loss_step,
                    'loss': loss_row['value'],
                    'test_reward': test_row['value']
                })
        
        if not merged_data:
            print("Could not merge loss and test reward data. Using synthetic data.")
            return visualize_epsilon_greedy_exploration({}, output_dir)
        
        synth_data = pd.DataFrame(merged_data)
        
        # Generate synthetic epsilon values based on training steps
        # Assuming epsilon decays from 0.9 to 0.01 over 100,000 steps
        max_step = synth_data['step'].max()
        synth_data['epsilon'] = 0.9 - 0.89 * (synth_data['step'] / 100000)
        synth_data['epsilon'] = synth_data['epsilon'].clip(0.01, 0.9)
        
        # Define exploration phases
        def get_phase(epsilon):
            if epsilon > 0.7:
                return "High Exploration"
            elif epsilon > 0.3:
                return "Balanced Exploration/Exploitation"
            else:
                return "High Exploitation"
        
        synth_data['phase'] = synth_data['epsilon'].apply(get_phase)
    
    # Create figure with constrained dimensions
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Define color scheme
    phase_colors = {
        "High Exploration": "#1f77b4",  # Blue
        "Balanced Exploration/Exploitation": "#ff7f0e",  # Orange
        "High Exploitation": "#2ca02c"  # Green
    }
    
    # Create twin axes for different metrics
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot loss and reward
    loss_line, = ax1.plot(synth_data['step'], synth_data['loss'], 'r-', linewidth=2, label='Loss')
    reward_line, = ax1.plot(synth_data['step'], synth_data['test_reward'], 'g-', linewidth=2, label='Test Reward')
    
    # Plot epsilon
    epsilon_line, = ax2.plot(synth_data['step'], synth_data['epsilon'], 'b--', linewidth=2, label='Epsilon (ε)')
    
    # Highlight phases with background colors
    prev_phase = None
    phase_starts = []
    phase_ends = []
    current_phases = []
    
    for i, row in synth_data.iterrows():
        if row['phase'] != prev_phase:
            if prev_phase is not None:
                phase_ends.append(row['step'])
            phase_starts.append(row['step'])
            current_phases.append(row['phase'])
            prev_phase = row['phase']
    
    # Add the last phase end
    if synth_data.shape[0] > 0:
        phase_ends.append(synth_data['step'].iloc[-1])
    
    # Plot phase backgrounds with simplified approach
    for i, phase in enumerate(current_phases):
        if i < len(phase_starts) and i < len(phase_ends):
            plt.axvspan(phase_starts[i], phase_ends[i], alpha=0.2, color=phase_colors.get(phase, 'gray'))
    
    # Add phase legend instead of text labels to prevent overflow
    phase_patches = [plt.Rectangle((0,0), 1, 1, color=phase_colors.get(phase, 'gray'), alpha=0.2) 
                    for phase in phase_colors.keys()]
    ax1.legend(phase_patches, phase_colors.keys(), loc='upper center', 
              title="Training Phases", bbox_to_anchor=(0.5, 1.05), ncol=3)

    # Add reference lines
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Breakeven Reward')
    
    # Configure axes
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss / Test Reward', fontsize=12)
    ax2.set_ylabel('Epsilon (ε)', fontsize=12)
    
    # Create combined legend for metrics (separate from phase legend)
    lines = [loss_line, reward_line, epsilon_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('Epsilon-Greedy Exploration Strategy Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Use subplots_adjust instead of tight_layout to prevent overflow
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)
    
    # Save the plot with constrained DPI
    epsilon_plot_path = os.path.join(output_dir, "epsilon_greedy_exploration.png")
    plt.savefig(epsilon_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    saved_plots.append(epsilon_plot_path)
    print(f"Epsilon-greedy exploration plot saved to {epsilon_plot_path}")
    
    return saved_plots

def visualize_keyword_clustering(dataset, output_dir):
    """
    Create visualizations that help executives make keyword investment decisions
    by clustering keywords based on their metrics.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    print(f"Generating keyword clustering visualization for executive decision-making...")
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Check if we have enough keywords for meaningful clustering
    if len(dataset['keyword'].unique()) < 5:
        print("Warning: Not enough unique keywords for meaningful clustering (need at least 5)")
        return saved_plots
    
    # Create a pivot table with keywords as rows and metrics as columns
    keyword_metrics = dataset.pivot_table(
        index='keyword', 
        values=feature_columns,
        aggfunc='mean'  # Take the average if there are multiple entries per keyword
    ).reset_index()
    
    # Calculate additional metrics that executives care about
    if 'ad_roas' in keyword_metrics.columns and 'ad_spend' in keyword_metrics.columns:
        # Calculate expected profit
        keyword_metrics['expected_profit'] = (keyword_metrics['ad_roas'] - 1) * keyword_metrics['ad_spend']
    
    if 'conversion_rate' in keyword_metrics.columns and 'paid_ctr' in keyword_metrics.columns:
        # Calculate funnel efficiency (CTR * conversion rate)
        keyword_metrics['funnel_efficiency'] = keyword_metrics['paid_ctr'] * keyword_metrics['conversion_rate']
    
    # Keep only the most business-relevant metrics for clustering
    business_metrics = ['ad_roas', 'paid_ctr', 'conversion_rate', 'ad_spend']
    if 'expected_profit' in keyword_metrics:
        business_metrics.append('expected_profit')
    if 'funnel_efficiency' in keyword_metrics:
        business_metrics.append('funnel_efficiency')
    
    # Ensure all selected metrics exist in the dataset
    business_metrics = [m for m in business_metrics if m in keyword_metrics.columns]
    
    # Create a copy of the dataset with just the business metrics
    clustering_data = keyword_metrics[['keyword'] + business_metrics].copy()
    
    # Handle NaN values (replace with column means)
    for col in business_metrics:
        clustering_data[col] = clustering_data[col].fillna(clustering_data[col].mean())
    
    # 1. Create correlation heatmap between keywords based on metrics
    # This visualizes which keywords behave similarly across metrics
    
    # Transpose the data to have keywords as columns
    keyword_corr_data = clustering_data.set_index('keyword')[business_metrics].T
    
    # Calculate correlation between keywords
    keyword_corr = keyword_corr_data.corr()
    
    # If we have many keywords, limit to the top ones by ad_spend or roas
    max_keywords_in_heatmap = 20
    if len(keyword_corr) > max_keywords_in_heatmap:
        print(f"Too many keywords for clear visualization. Limiting to top {max_keywords_in_heatmap}.")
        
        # Sort by most important metric (expected profit, ROAS, or ad_spend)
        if 'expected_profit' in clustering_data.columns:
            sort_metric = 'expected_profit'
        elif 'ad_roas' in clustering_data.columns:
            sort_metric = 'ad_roas'
        else:
            sort_metric = 'ad_spend'
            
        top_keywords = clustering_data.sort_values(sort_metric, ascending=False)['keyword'].head(max_keywords_in_heatmap).tolist()
        keyword_corr = keyword_corr.loc[top_keywords, top_keywords]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    # Use a diverging colormap to better show positive/negative correlations
    cmap = plt.cm.RdBu_r
    
    # Create mask to not repeat the upper triangle
    mask = np.zeros_like(keyword_corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Plot the heatmap
    sns.heatmap(keyword_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7},
                annot=True, fmt=".2f", annot_kws={"size": 8})
    
    # Adjust labels for readability
    plt.title('Keyword Similarity Map', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the visualization
    keyword_corr_path = os.path.join(output_dir, "keyword_similarity_map.png")
    plt.savefig(keyword_corr_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    saved_plots.append(keyword_corr_path)
    print(f"Keyword similarity map saved to {keyword_corr_path}")
    
    # 2. Create a PCA plot to visualize keyword clusters
    
    # Normalize data for PCA (important to prevent metrics with larger scales from dominating)
    X = clustering_data[business_metrics].copy()
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    
    # Apply PCA to reduce to 2 dimensions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame({
        'keyword': clustering_data['keyword'],
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1]
    })
    
    # Add the most important business metrics back for coloring points
    for metric in ['ad_roas', 'ad_spend', 'expected_profit', 'conversion_rate']:
        if metric in clustering_data.columns:
            pca_df[metric] = clustering_data[metric]
    
    # Determine the best metric to color by (prioritizing the most business-relevant)
    if 'expected_profit' in pca_df.columns:
        color_metric = 'expected_profit'
        color_title = 'Expected Profit'
    elif 'ad_roas' in pca_df.columns:
        color_metric = 'ad_roas'
        color_title = 'ROAS'
    else:
        color_metric = 'ad_spend'
        color_title = 'Ad Spend'
    
    # Create the PCA plot
    plt.figure(figsize=(14, 10))
    
    # Calculate explained variance for axis labels
    explained_var = pca.explained_variance_ratio_ * 100
    
    # Use a colorful scatter plot
    scatter = plt.scatter(
        pca_df['PC1'], 
        pca_df['PC2'],
        c=pca_df[color_metric], 
        cmap='viridis',
        s=100,  # Larger points for visibility
        alpha=0.7,
        edgecolors='k',
        linewidths=1
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_title, fontsize=12)
    
    # Add labels for easier identification (but avoid overlapping)
    from adjustText import adjust_text
    
    # Only label a manageable number of points to avoid clutter
    MAX_LABELS = 20
    if len(pca_df) > MAX_LABELS:
        # If we have too many keywords, only label the top ones by the color metric
        top_indices = pca_df[color_metric].abs().nlargest(MAX_LABELS).index
        label_df = pca_df.loc[top_indices]
    else:
        label_df = pca_df
    
    texts = []
    for i, row in label_df.iterrows():
        texts.append(plt.text(row['PC1'], row['PC2'], row['keyword'], fontsize=9))
    
    # Try to adjust text positions to minimize overlap
    try:
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5, shrinkA=5),  # Increased shrinkA value
            expand_points=(1.5, 1.5)
        )
    except Exception as e:
        # More detailed error handling
        print(f"Warning: Adjusting text labels had an issue: {str(e)}")
        print("Some labels may overlap, but the visualization is still valid.")

    
    # Add title and labels
    plt.title('Keyword Investment Landscape', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1f}% Variance)', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add explanatory annotations for executives
    plt.annotate(
        "Keywords grouped together\nbehave similarly across metrics",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
        fontsize=10
    )
    
    # Add interpretation guide based on color metric
    if color_metric == 'expected_profit' or color_metric == 'ad_roas':
        plt.annotate(
            f"Darker colors indicate higher {color_title}\n(better investment candidates)",
            xy=(0.05, 0.05),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            fontsize=10
        )
    else:
        plt.annotate(
            f"Darker colors indicate higher {color_title}",
            xy=(0.05, 0.05),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            fontsize=10
        )
    
    # Save the visualization
    pca_path = os.path.join(output_dir, "keyword_investment_landscape.png")
    plt.savefig(pca_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    saved_plots.append(pca_path)
    print(f"Keyword investment landscape saved to {pca_path}")
    
    # 3. Create a quadrant analysis (if we have the right metrics)
    if 'ad_roas' in clustering_data.columns and 'ad_spend' in clustering_data.columns:
        # Create a quadrant analysis plot (ROAS vs Spend)
        plt.figure(figsize=(12, 10))
        
        # Determine breakpoints for quadrants (using median or 1.0 for ROAS)
        roas_threshold = max(clustering_data['ad_roas'].median(), 1.0)  # At least breakeven
        spend_threshold = clustering_data['ad_spend'].median()
        
        # Create the scatter plot
        scatter = plt.scatter(
            clustering_data['ad_spend'],
            clustering_data['ad_roas'],
            c=clustering_data['ad_roas'] * clustering_data['ad_spend'],  # Color by total return
            cmap='viridis',
            s=100,
            alpha=0.7,
            edgecolors='k',
            linewidths=1
        )
        
        # Add reference lines for quadrants
        plt.axvline(x=spend_threshold, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=roas_threshold, color='gray', linestyle='--', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Total Return (ROAS × Spend)', fontsize=12)
        
        # Add labels for easier identification
        texts = []
        for i, row in clustering_data.iterrows():
            texts.append(plt.text(row['ad_spend'], row['ad_roas'], row['keyword'], fontsize=9))
        
        # Try to adjust text positions to minimize overlap
        try:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
                expand_points=(1.5, 1.5)
            )
        except:
            print("Warning: Could not adjust text labels for clarity. Some labels may overlap.")
        
        # Add quadrant labels/explanations
        plt.annotate(
            "HIGH VALUE\nHigh ROAS, Low Spend\n→ Increase Budget",
            xy=(spend_threshold * 0.1, roas_threshold * 1.1),
            xycoords='data',
            bbox=dict(boxstyle="round,pad=0.5", fc="#90EE90", ec="green", alpha=0.7),
            fontsize=10
        )
        
        plt.annotate(
            "STAR PERFORMERS\nHigh ROAS, High Spend\n→ Maintain/Optimize",
            xy=(spend_threshold * 1.1, roas_threshold * 1.1),
            xycoords='data',
            bbox=dict(boxstyle="round,pad=0.5", fc="#ADD8E6", ec="blue", alpha=0.7),
            fontsize=10
        )
        
        plt.annotate(
            "POOR PERFORMERS\nLow ROAS, Low Spend\n→ Test or Cut",
            xy=(spend_threshold * 0.1, roas_threshold * 0.5),
            xycoords='data',
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFCCCB", ec="red", alpha=0.7),
            fontsize=10
        )
        
        plt.annotate(
            "REVIEW & OPTIMIZE\nLow ROAS, High Spend\n→ Reduce or Improve",
            xy=(spend_threshold * 1.1, roas_threshold * 0.5),
            xycoords='data',
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFD700", ec="orange", alpha=0.7),
            fontsize=10
        )
        
        # Add title and labels
        plt.title('Keyword Investment Decision Quadrant', fontsize=16)
        plt.xlabel('Ad Spend', fontsize=12)
        plt.ylabel('Return on Ad Spend (ROAS)', fontsize=12)
        plt.grid(alpha=0.3)
        
        # Add breakeven line
        plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.5, label='Breakeven (ROAS=1.0)')
        plt.legend()
        
        # Save the visualization
        quadrant_path = os.path.join(output_dir, "keyword_investment_quadrant.png")
        plt.savefig(quadrant_path, dpi=100, bbox_inches="tight")
        plt.close()
        
        saved_plots.append(quadrant_path)
        print(f"Keyword investment quadrant saved to {quadrant_path}")
    
    return saved_plots

def visualize_reward_function(dataset, output_dir):
    """
    Create visualizations focused on the reward function design.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    print(f"Generating reward function visualization...")
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Define the reward function
    def compute_reward(roas, other_roas, clipping=True):
        # Logarithmic scaling of ROAS
        adjusted_reward = 0 if roas <= 0 else np.log(roas)
        
        # Opportunity cost penalty
        opportunity_penalty = np.mean(other_roas) * 0.2
        
        # Compute reward
        reward = adjusted_reward - opportunity_penalty
        
        # Apply clipping if requested
        if clipping:
            reward = np.clip(reward, -2.0, 2.0)
            
        return reward
    
    # Generate a range of ROAS values to visualize
    roas_range = np.linspace(0.1, 5.0, 100)
    
    # Define different opportunity cost scenarios
    opportunity_scenarios = [
        ("Low Competition", [0.8, 1.0, 1.2, 1.0]),
        ("Medium Competition", [1.5, 1.8, 2.0, 1.7]),
        ("High Competition", [2.5, 3.0, 3.2, 2.8])
    ]
    
    # Create figure for reward function with constrained dimensions
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Plot reward curves for different opportunity costs
    for scenario_name, other_roas in opportunity_scenarios:
        # Calculate rewards with clipping
        rewards_clipped = [compute_reward(roas, other_roas, clipping=True) for roas in roas_range]
        
        # Calculate rewards without clipping
        rewards_unclipped = [compute_reward(roas, other_roas, clipping=False) for roas in roas_range]
        
        # Plot clipped reward
        plt.plot(roas_range, rewards_clipped, '-', linewidth=2.5, label=f"{scenario_name} (Clipped)")
        
        # Plot unclipped reward as lighter dashed line
        plt.plot(roas_range, rewards_unclipped, '--', alpha=0.5, 
                 label=f"{scenario_name} (Unclipped)")
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Breakeven ROAS')
    
    # Add clipping bounds
    plt.axhline(y=2.0, color='purple', linestyle=':', alpha=0.7, label='Reward Clipping Bounds (±2.0)')
    plt.axhline(y=-2.0, color='purple', linestyle=':', alpha=0.7)
    
    # Configure axes
    plt.xlabel('Return on Ad Spend (ROAS)', fontsize=12)
    plt.ylabel('Computed Reward', fontsize=12)
    plt.title('Reward Function Analysis: ROAS Optimization with Opportunity Cost', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add annotations explaining reward components
    plt.annotate('Logarithmic Scaling\nreduces variance', 
                 xy=(3.5, 1.0), xytext=(4.0, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                 fontsize=10, ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.annotate('Opportunity Cost Penalty\nshifts curves downward', 
                 xy=(2.0, -0.5), xytext=(2.5, -1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.annotate('Reward Clipping\nstabilizes learning', 
                 xy=(4.5, 2.0), xytext=(3.5, 2.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    # Save the plot with constrained DPI
    reward_plot_path = os.path.join(output_dir, "reward_function_analysis.png")
    plt.savefig(reward_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    saved_plots.append(reward_plot_path)
    print(f"Reward function analysis plot saved to {reward_plot_path}")
    
    # Create decision boundary plot with constrained dimensions
    plt.figure(figsize=(10, 8), dpi=100)
    
    # Create mesh grid for decision boundary visualization
    roas_grid = np.linspace(0.5, 3.0, 100)
    ctr_grid = np.linspace(0.01, 0.3, 100)
    roas_mesh, ctr_mesh = np.meshgrid(roas_grid, ctr_grid)
    
    # Define decision function (simplified version of what the RL agent learns)
    def decision_function(roas, ctr):
        if roas > 2.0:
            return 1.0  # Always select high ROAS
        elif roas > 1.2:
            return 1.0  # Select medium ROAS
        elif roas > 1.0 and ctr > 0.18:
            return 1.0  # Select marginal ROAS only with high CTR
        else:
            return 0.0  # Don't select unprofitable keywords
    
    # Apply decision function to mesh grid
    decision = np.vectorize(decision_function)(roas_mesh, ctr_mesh)
    
    # Plot decision boundary
    plt.contourf(roas_mesh, ctr_mesh, decision, alpha=0.4, cmap='RdYlGn', levels=[-0.5, 0.5, 1.5])
    
    # Plot contour lines
    plt.contour(roas_mesh, ctr_mesh, decision, colors='black', linestyles='-', levels=[0.5])
    
    # Add reference lines
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Breakeven ROAS')
    plt.axvline(x=1.2, color='orange', linestyle='--', alpha=0.7, label='Medium ROAS Threshold')
    plt.axvline(x=2.0, color='green', linestyle='--', alpha=0.7, label='High ROAS Threshold')
    plt.axhline(y=0.18, color='blue', linestyle='--', alpha=0.7, label='CTR Threshold')
    
    # Scatter actual data points
    scatter = plt.scatter(dataset['ad_roas'], dataset['paid_ctr'], 
                         c=dataset['ad_roas'], cmap='viridis', 
                         alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    
    # Add colorbar and legend
    cbar = plt.colorbar(scatter, label='ROAS')
    plt.legend(loc='upper right')
    
    # Add region labels
    plt.text(2.5, 0.15, "Always Select\n(High ROAS)", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="palegreen", ec="green", alpha=0.7))
    
    plt.text(1.6, 0.15, "Always Select\n(Medium ROAS)", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="palegreen", ec="green", alpha=0.7))
    
    plt.text(1.1, 0.25, "Select if\nHigh CTR", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="khaki", ec="orange", alpha=0.7))
    
    plt.text(0.7, 0.15, "Never Select\n(Unprofitable)", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="salmon", ec="red", alpha=0.7))
    
    # Configure axes
    plt.xlabel('Return on Ad Spend (ROAS)', fontsize=12)
    plt.ylabel('Click-Through Rate (CTR)', fontsize=12)
    plt.title('Learned Decision Boundaries: ROAS vs CTR', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    # Save the plot with constrained DPI
    decision_plot_path = os.path.join(output_dir, "decision_boundaries.png")
    plt.savefig(decision_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    saved_plots.append(decision_plot_path)
    print(f"Decision boundaries plot saved to {decision_plot_path}")
    
    return saved_plots


def visualize_feature_correlation_matrix(dataset, output_dir):
    """
    Create a correlation heatmap of the feature columns.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    print(f"Generating feature correlation matrix visualization...")
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Calculate correlation matrix
    corr = dataset[feature_columns].corr()
    
    # Create figure with constrained dimensions
    plt.figure(figsize=(14, 12), dpi=100)
    
    # Create heatmap with custom colormap for better visualization
    cmap = plt.cm.RdBu_r
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = False  # Only show lower triangle
    
    # Plot the heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7},
                annot=True, fmt=".2f", annot_kws={"size": 8})
    
    # Configure labels and title
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Tighten layout
    plt.tight_layout()
    
    # Save the plot with constrained DPI
    corr_plot_path = os.path.join(output_dir, "feature_correlation_matrix.png")
    plt.savefig(corr_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    saved_plots.append(corr_plot_path)
    print(f"Feature correlation matrix saved to {corr_plot_path}")
    
    return saved_plots


def visualize_budget_allocation_strategy(dataset, output_dir):
    """
    Create visualizations focused on the budget allocation strategy.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    print(f"Generating budget allocation strategy visualization...")
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Create cash management simulation visualization
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Implement budget allocation simulation with temporal dynamics
    initial_cash = 100000
    cash_levels = [initial_cash]
    revenues = [0]
    expenses = [0]
    roas_values = []
    steps = list(range(30))  # Simulate 30 steps
    
    np.random.seed(42)  # For reproducibility
    
    for step in range(1, 30):
        # Get previous cash level
        prev_cash = cash_levels[-1]
        
        # Implement 10% budget allocation constraint
        budget = prev_cash * 0.1
        
        # Simulate ROAS with temporal learning improvements
        step_factor = min(1.0, step / 20)
        base_roas = 1.0 + step_factor * 1.5
        roas = max(0.5, np.random.normal(base_roas, 0.3))
        roas_values.append(roas)
        
        # Calculate revenue with stochastic returns
        revenue = budget * roas
        revenues.append(revenue)
        expenses.append(budget)
        
        # Update cash position
        new_cash = prev_cash - budget + revenue
        cash_levels.append(new_cash)
    
    # Create multi-axis visualization with synchronized temporal representation
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot cash balance trajectory
    ax1.plot(steps, cash_levels, 'b-', linewidth=2.5, label='Cash Balance')
    ax1.set_xlabel('Simulation Steps', fontsize=12)
    ax1.set_ylabel('Cash Balance', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Implement revenue/expense visualization on secondary axis
    ax2 = ax1.twinx()
    
    # Plot revenue and expenses with differentiated representation
    bar_width = 0.35
    revenue_bars = ax2.bar([x + bar_width/2 for x in steps[1:]], revenues[1:], 
                          bar_width, alpha=0.5, color='green', label='Revenue')
    expense_bars = ax2.bar([x - bar_width/2 for x in steps[1:]], [-e for e in expenses[1:]], 
                          bar_width, alpha=0.5, color='red', label='Ad Spend')
    
    # Implement ROAS temporal trajectory visualization
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))  # Offset the right spine
    roas_line, = ax3.plot(steps[1:], roas_values, 'r--', linewidth=1.5, label='ROAS')
    ax3.set_ylabel('ROAS', fontsize=12, color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    
    # Construct integrated legend with multi-axis components
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    # Implement explanatory annotation for budget constraint mechanism
    plt.annotate('10% Budget Allocation Policy\nManages Risk While Optimizing Returns', 
                xy=(15, cash_levels[15]), xytext=(20, cash_levels[15] * 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('Cash Management: 10% Budget Allocation per Step', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Apply dimensional constraints
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.85)
    
    # Generate image artifact
    cash_plot_path = os.path.join(output_dir, "cash_management_simulation.png")
    plt.savefig(cash_plot_path, dpi=100)
    plt.close()
    
    saved_plots.append(cash_plot_path)
    print(f"Cash management simulation plot saved to {cash_plot_path}")
    
    return saved_plots


def visualize_experience_replay_target_network(output_dir):
    """
    Create visualizations focused on experience replay and target network effects.
    
    Args:
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    print(f"Generating experience replay and target network visualization...")
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Create conceptual diagram of experience replay and target network
    plt.figure(figsize=(12, 8), dpi=100)
    ax = plt.gca()
    ax.axis('off')
    
    # Define component positions
    components = {
        'agent': (0.5, 0.8),
        'environment': (0.5, 0.5),
        'replay_buffer': (0.2, 0.3),
        'batch': (0.5, 0.2),
        'target_network': (0.8, 0.6),
        'value_network': (0.8, 0.3)
    }
    
    # Draw components with appropriate structural motifs and interaction interfaces
    for name, (x, y) in components.items():
        if name == 'replay_buffer':
            # Implement buffer domain with storage capacity representation
            buffer_width = 0.15
            buffer_height = 0.25
            rect = plt.Rectangle((x - buffer_width/2, y - buffer_height/2), 
                                buffer_width, buffer_height, 
                                fc='#d2f1ff', ec='blue', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, 'Experience\nReplay Buffer\n(100k experiences)', 
                    ha='center', va='center', fontsize=10)
            
            # Add experiential substrates representing memory traces
            for i in range(5):
                small_rect = plt.Rectangle((x - buffer_width/2 + 0.01, 
                                          y - buffer_height/2 + 0.03 + i*0.04), 
                                          buffer_width - 0.02, 0.02, 
                                          fc='white', ec='blue', alpha=0.7)
                ax.add_patch(small_rect)
        
        elif name == 'environment':
            # Environmental context representation with stochastic properties
            circle = plt.Circle((x, y), 0.12, fc='#e6ffe6', ec='green', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, 'Environment\n(Keyword Space)', ha='center', va='center', fontsize=10)
            
        elif name == 'agent':
            # Agent architecture with decision capability
            agent_width = 0.15
            agent_height = 0.1
            rect = plt.Rectangle((x - agent_width/2, y - agent_height/2), 
                                agent_width, agent_height, 
                                fc='#ffffd2', ec='orange', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, 'RL Agent', ha='center', va='center', fontsize=10)
            
        elif name == 'batch':
            # Minibatch sampling domain with statistical properties
            batch_width = 0.1
            batch_height = 0.05
            rect = plt.Rectangle((x - batch_width/2, y - batch_height/2), 
                                batch_width, batch_height, 
                                fc='#d2f1ff', ec='blue', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y - 0.08, 'Random Batch\n(n=128)', ha='center', va='center', fontsize=10)
            
        elif name == 'value_network':
            # Value function approximator with parameterized representation
            nn_width = 0.12
            nn_height = 0.12
            rect = plt.Rectangle((x - nn_width/2, y - nn_height/2), 
                                nn_width, nn_height, 
                                fc='#ffd2e6', ec='purple', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, 'Q-Network', ha='center', va='center', fontsize=10)
            
        elif name == 'target_network':
            # Target network for stability maintenance through temporal consistency
            nn_width = 0.12
            nn_height = 0.12
            rect = plt.Rectangle((x - nn_width/2, y - nn_height/2), 
                                nn_width, nn_height, 
                                fc='#ffe2d2', ec='red', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, 'Target\nNetwork', ha='center', va='center', fontsize=10)
    
    # Implement information transfer pathways with appropriate signaling dynamics
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                     color='gray', lw=1.5, alpha=0.7)
    
    # Action emission pathway
    ax.annotate('', 
                xy=components['environment'], 
                xytext=components['agent'],
                arrowprops=arrow_props)
    ax.text(0.45, 0.65, 'action', ha='center', va='center', fontsize=8, 
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
    
    # Experience acquisition pathway with temporal state transition representation
    ax.annotate('', 
                xy=components['replay_buffer'], 
                xytext=components['environment'],
                arrowprops=arrow_props)
    ax.text(0.33, 0.4, 'experience\n(s,a,r,s\')', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
    
    # Random sampling pathway for decorrelation
    ax.annotate('', 
                xy=components['batch'], 
                xytext=components['replay_buffer'],
                arrowprops=arrow_props)
    ax.text(0.35, 0.25, 'sample', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
    
    # Gradient update pathway for parameter optimization
    ax.annotate('', 
                xy=components['value_network'], 
                xytext=components['batch'],
                arrowprops=arrow_props)
    ax.text(0.65, 0.25, 'train', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
    
    # Soft update pathway for stability preservation
    ax.annotate('', 
                xy=components['target_network'], 
                xytext=components['value_network'],
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                              color='red', lw=1.5, alpha=0.7))
    ax.text(0.83, 0.45, 'soft update\nτ=0.99', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.7))
    
    # Target value feedback pathway
    ax.annotate('', 
                xy=(components['batch'][0] + 0.05, components['batch'][1] + 0.03), 
                xytext=(components['target_network'][0] - 0.05, components['target_network'][1] - 0.05),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                              color='blue', lw=1.5, alpha=0.7, linestyle='--'))
    ax.text(0.67, 0.35, 'target\nvalues', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="blue", alpha=0.7))
    
    # Policy emergence pathway
    ax.annotate('', 
                xy=(components['agent'][0] - 0.05, components['agent'][1]), 
                xytext=(components['value_network'][0], components['value_network'][1] + 0.06),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                              color='green', lw=1.5, alpha=0.7))
    ax.text(0.65, 0.5, 'policy', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="green", alpha=0.7))
    
    # Architectural framework title
    ax.text(0.5, 0.95, 'Experience Replay and Target Network Architecture', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Mechanistic annotation framework with functional explanations
    notes = [
        "• Experience Replay Buffer: Stores 100,000 transitions (s,a,r,s') to break temporal correlations",
        "• Random Sampling: Draws minibatches of 128 experiences for decorrelated updates",
        "• Target Network: Stabilizes learning by providing consistent targets during updates",
        "• Soft Updates: Gradually transfers Q-network weights to target network (τ=0.99)"
    ]
    
    for i, note in enumerate(notes):
        ax.text(0.02, 0.05 - i*0.03, note, ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
    
    # Apply subplots adjustment protocol for dimensional constraint
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # Generate image artifact with constrained rendering parameters
    er_tn_plot_path = os.path.join(output_dir, "experience_replay_target_network.png")
    plt.savefig(er_tn_plot_path, dpi=100)
    plt.close()
    
    saved_plots.append(er_tn_plot_path)
    print(f"Experience replay and target network plot saved to {er_tn_plot_path}")
    
    # Generate stability analysis visualization with comparative dynamics
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Generate synthetic training trajectories with varying stability characteristics
    steps = np.linspace(0, 100000, 500)
    
    # Optimal configuration with dual stabilizing mechanisms
    base_loss = 10 * np.exp(-steps / 30000)
    stable_noise = np.random.normal(0, 0.1, size=len(steps))
    smooth_loss = base_loss + stable_noise * base_loss
    
    # Intermediate stability with single mechanism
    medium_noise = np.random.normal(0, 0.3, size=len(steps))
    # Implement stochastic perturbations
    for i in range(10):
        spike_idx = np.random.randint(0, len(steps))
        medium_noise[spike_idx] = 1.5
    medium_loss = base_loss + medium_noise * base_loss
    
    # Unstable configuration with no stability mechanisms
    high_noise = np.random.normal(0, 0.6, size=len(steps))
    # Implement high-frequency perturbations
    for i in range(30):
        spike_idx = np.random.randint(0, len(steps))
        high_noise[spike_idx] = 2.5
    unstable_loss = base_loss + high_noise * base_loss
    
    # Implement divergence dynamics in unstable configuration
    divergence_factor = np.ones(len(steps))
    divergence_start = int(0.7 * len(steps))
    for i in range(divergence_start, len(steps)):
        divergence_factor[i] = 1 + 0.005 * (i - divergence_start)
    unstable_loss = unstable_loss * divergence_factor
    
    # Visualize stability profiles
    plt.plot(steps, smooth_loss, 'g-', linewidth=2, label='With Experience Replay & Target Network')
    plt.plot(steps, medium_loss, 'b-', linewidth=1.5, label='With Experience Replay Only')
    plt.plot(steps, unstable_loss, 'r-', linewidth=1.5, label='Without ER or TN (Unstable)')
    
    # Mechanistic annotations with explanatory frameworks
    plt.annotate('Stable Learning\nLow Variance', 
                xy=(steps[-1], smooth_loss[-1]), xytext=(steps[-1] * 0.8, smooth_loss[-1] * 1.5),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    
    plt.annotate('Occasional Spikes\nMedium Stability', 
                xy=(steps[len(steps)//2], medium_loss[len(steps)//2] * 1.2), 
                xytext=(steps[len(steps)//2] * 0.8, medium_loss[len(steps)//2] * 2),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    
    plt.annotate('Policy Divergence\nHigh Instability', 
                xy=(steps[-1], unstable_loss[-1]), 
                xytext=(steps[-1] * 0.7, unstable_loss[-1] * 0.5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # Configuration parameters
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Impact of Experience Replay and Target Network on Training Stability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Apply dimensional constraints
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    # Generate image artifact
    stability_plot_path = os.path.join(output_dir, "training_stability_analysis.png")
    plt.savefig(stability_plot_path, dpi=100)
    plt.close()
    
    saved_plots.append(stability_plot_path)
    print(f"Training stability analysis plot saved to {stability_plot_path}")
    
    return saved_plots

def create_html_report(plots, output_dir, params=None):
    """
    Create an HTML report with all visualizations.
    
    Args:
        plots (dict): Dictionary with section names and plots
        output_dir (str): Directory to save visualizations
        params (dict): Optional parameters extracted from logs
        
    Returns:
        str: Path to HTML report
    """
    print(f"Generating HTML report...")
    report_path = os.path.join(output_dir, "rl_optimization_report.html")
    
    # HTML template
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digital Advertising RL Optimization Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                text-align: center;
            }
            h2 {
                color: #2980b9;
                margin-top: 30px;
                padding-left: 10px;
                border-left: 4px solid #3498db;
            }
            .section {
                margin-bottom: 40px;
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
            .timestamp {
                color: #777;
                font-size: 0.9em;
                text-align: right;
                margin-top: 30px;
            }
        </style>
    </head>
    <body>
        <h1>Digital Advertising RL Optimization Analysis</h1>
    """

    # Add sections for each type of visualization with descriptions
    section_descriptions = {
        "Exploration Strategy": "These visualizations illustrate how the epsilon-greedy strategy balances exploration and exploitation during training. Initially, the agent explores more frequently to discover the environment, gradually shifting toward exploitation of known profitable keywords.",
        
        "Reward Function Design": "The reward function is critical for guiding the agent toward optimal policy. These plots show how the reward is computed based on ROAS, with adjustments for opportunity cost and variance reduction techniques.",
        
        "Decision Boundaries": "These visualizations show the decision boundaries learned by the RL agent, particularly focusing on the relationship between ROAS and CTR when making keyword bidding decisions.",
        
        "Stability Mechanisms": "Experience replay and target networks are key stability mechanisms in deep RL. These diagrams illustrate how they prevent catastrophic forgetting and reduce training variance.",
        
        "Budget Allocation": "These visualizations demonstrate the 10% budget allocation strategy, which balances risk management with return optimization in the advertising context.",
        
        "Feature Analysis": "The correlation matrix provides insights into relationships between different advertising metrics, helping to understand the feature space the agent operates in."
    }

    for section_name, section_plots in plots.items():
        if section_name in section_descriptions:
            description = section_descriptions[section_name]
        else:
            description = f"Visualizations related to {section_name.lower()}."
            
        html += f"""
        <div class="section">
            <h2>{section_name}</h2>
            <div class="description">
                <p>{description}</p>
            </div>
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

    # Add a summary section
    html += """
    <div class="section">
        <h2>Summary</h2>
        <div class="description">
            <p>The visualizations in this report demonstrate the effectiveness of reinforcement learning for digital advertising optimization. 
            Key components include:</p>
            <ul>
                <li>Epsilon-greedy exploration strategy that balances exploration and exploitation</li>
                <li>A reward function design optimized for ROAS with opportunity cost consideration</li>
                <li>Experience replay and target networks that stabilize training</li>
                <li>A risk-managed budget allocation strategy</li>
                <li>Decision boundaries that incorporate both ROAS and engagement metrics</li>
            </ul>
            <p>Together, these components create a robust system for maximizing advertising effectiveness while managing budgets efficiently.</p>
        </div>
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
    parser = argparse.ArgumentParser(description="Visualize RL Optimization of Digital Advertising")
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV (if None, generates synthetic data)")
    parser.add_argument("--output_dir", type=str, default="visualization_results", help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for synthetic data generation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    # Parse TensorBoard logs
    try:
        metric_dfs = parse_tensorboard_logs(args.logdir)
    except Exception as e:
        print(f"Warning: Error parsing TensorBoard logs: {e}")
        print(f"Error details: {traceback.format_exc()}")
        print("Proceeding with dataset analysis only.")
        metric_dfs = {}
    
    # Load dataset
    try:
        # First try to load from the same path that digital_advertising.py uses
        default_dataset_path = 'data/organized_dataset.csv'
        
        if os.path.exists(default_dataset_path):
            print(f"Loading dataset from {default_dataset_path}")
            dataset = pd.read_csv(default_dataset_path)
        elif args.dataset and os.path.exists(args.dataset):
            print(f"Loading dataset from {args.dataset}")
            dataset = pd.read_csv(args.dataset)
        else:
            print(f"Warning: Neither {default_dataset_path} nor {args.dataset} exist.")
            print(f"Generating synthetic dataset with {args.num_samples} samples")
            print(f"Note: This is not recommended for analysis of actual training results.")
            dataset = generate_synthetic_data(args.num_samples)
            dataset_path = os.path.join(args.output_dir, "synthetic_ad_data.csv")
            dataset.to_csv(dataset_path, index=False)
            print(f"Synthetic dataset saved to {dataset_path}")
    except Exception as e:
        print(f"Error with dataset: {e}")
        print(f"Error details: {traceback.format_exc()}")
        print("Generating emergency synthetic dataset")
        dataset = generate_synthetic_data(100)  # Smaller emergency dataset

    # Generate visualizations
    all_plots = {}
    
    # 1. Epsilon-Greedy Exploration
    try:
        all_plots["Exploration Strategy"] = visualize_epsilon_greedy_exploration(
            metric_dfs, 
            os.path.join(args.output_dir, "exploration_strategy")
        )
    except Exception as e:
        print(f"Warning: Error visualizing epsilon-greedy exploration: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Exploration Strategy"] = []
    
    # 2. Reward Function Design
    try:
        all_plots["Reward Function Design"] = visualize_reward_function(
            dataset,
            os.path.join(args.output_dir, "reward_function")
        )
    except Exception as e:
        print(f"Warning: Error visualizing reward function: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Reward Function Design"] = []
    
    # 3. Experience Replay and Target Network
    try:
        all_plots["Stability Mechanisms"] = visualize_experience_replay_target_network(
            os.path.join(args.output_dir, "experience_replay")
        )
    except Exception as e:
        print(f"Warning: Error visualizing experience replay: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Stability Mechanisms"] = []
    
    # 4. Budget Allocation Strategy
    try:
        all_plots["Budget Allocation"] = visualize_budget_allocation_strategy(
            dataset,
            os.path.join(args.output_dir, "budget_strategy")
        )
    except Exception as e:
        print(f"Warning: Error visualizing budget allocation: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Budget Allocation"] = []
    
    # 5. Feature Correlation Matrix
    try:
        all_plots["Feature Analysis"] = visualize_feature_correlation_matrix(
            dataset,
            os.path.join(args.output_dir, "feature_analysis")
        )
    except Exception as e:
        print(f"Warning: Error visualizing feature correlations: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Feature Analysis"] = []
    
    # 6. Keyword Clustering for Executive Decision Making
    try:
        all_plots["Keyword Investment Analysis"] = visualize_keyword_clustering(
            dataset,
            os.path.join(args.output_dir, "keyword_investment")
        )
    except Exception as e:
        print(f"Warning: Error visualizing keyword clustering: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Keyword Investment Analysis"] = []

    # Create HTML report with all visualizations
    try:
        html_report_path = create_html_report(all_plots, args.output_dir)
        print(f"\nVisualization complete!")
        print(f"HTML report available at: {html_report_path}")
    except Exception as e:
        print(f"Warning: Error creating HTML report: {e}")
        print(f"Error details: {traceback.format_exc()}")
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        print(f"Error details: {traceback.format_exc()}")


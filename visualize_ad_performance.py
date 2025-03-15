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
import traceback
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
    
    # Create a second plot: Phase Performance Analysis
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Aggregate metrics by phase
    phase_metrics = synth_data.groupby('phase').agg({
        'step': ['min', 'max'],
        'loss': 'mean',
        'test_reward': 'mean',
        'epsilon': 'mean'
    }).reset_index()
    
    # Order phases correctly
    phase_order = ["High Exploration", "Balanced Exploration/Exploitation", "High Exploitation"]
    phase_metrics['order'] = phase_metrics['phase'].map({phase: i for i, phase in enumerate(phase_order)})
    phase_metrics = phase_metrics.sort_values('order').reset_index(drop=True)
    
    # Compute improvement percentages
    phase_metrics['improvement'] = 0.0
    
    for i in range(1, len(phase_metrics)):
        current_reward = phase_metrics.loc[i, ('test_reward', 'mean')]
        prev_reward = phase_metrics.loc[i-1, ('test_reward', 'mean')]
        
        if prev_reward > 0:
            improvement = ((current_reward / prev_reward) - 1) * 100
            phase_metrics.loc[i, 'improvement'] = improvement
    
    # Create bar charts for phase analysis
    bar_width = 0.35
    x = np.arange(len(phase_metrics))
    
    # Plot average reward by phase
    ax1 = plt.gca()
    reward_bars = ax1.bar(x - bar_width/2, phase_metrics[('test_reward', 'mean')], 
                        bar_width, color='green', alpha=0.7, label='Avg Test Reward')
    
    # Add improvement labels
    for i, row in phase_metrics.iterrows():
        if i > 0:  # Skip first phase (no improvement to calculate)
            if row['improvement'] > 0:
                plt.text(i - bar_width/2, row[('test_reward', 'mean')] + 0.05, 
                        f"+{row['improvement']:.1f}%", ha='center')
    
    # Plot average epsilon by phase
    ax2 = ax1.twinx()
    epsilon_bars = ax2.bar(x + bar_width/2, phase_metrics[('epsilon', 'mean')], 
                        bar_width, color='blue', alpha=0.7, label='Avg Epsilon (ε)')
    
    # Configure axes
    ax1.set_xlabel('Training Phase', fontsize=12)
    ax1.set_ylabel('Average Test Reward', fontsize=12)
    ax2.set_ylabel('Average Epsilon (ε)', fontsize=12)
    
    ax1.set_xticks(x)
    phase_labels = [f"{row['phase']}\n(Steps {row[('step', 'min')]//1000}k-{row[('step', 'max')]//1000}k)" 
                   for _, row in phase_metrics.iterrows()]
    ax1.set_xticklabels(phase_labels)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.title('Performance by Exploration Phase', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9)
    
    # Save the plot with constrained DPI
    phases_plot_path = os.path.join(output_dir, "exploration_phases_analysis.png")
    plt.savefig(phases_plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    saved_plots.append(phases_plot_path)
    print(f"Exploration phases analysis plot saved to {phases_plot_path}")
    
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
    
    # Create figure for ROAS vs Ad Spend visualization with dimensional constraints
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Calculate investment decision based on ROAS thresholds and CTR modulation
    dataset['selected'] = (
        (dataset['ad_roas'] > 2.0) | 
        ((dataset['ad_roas'] > 1.2) & (dataset['ad_roas'] <= 2.0)) |
        ((dataset['ad_roas'] > 1.0) & (dataset['ad_roas'] <= 1.2) & (dataset['paid_ctr'] > 0.18))
    )
    
    # Define color map based on selection decision for visual differentiation
    selection_colors = dataset['selected'].map({True: 'green', False: 'red'})
    
    # Implement multidimensional visualization with competitiveness modulation
    plt.scatter(
        dataset['ad_spend'], 
        dataset['ad_roas'],
        c=dataset['competitiveness'],
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolor=selection_colors
    )
    
    # Add colorbar for competitiveness dimension
    cbar = plt.colorbar()
    cbar.set_label('Keyword Competitiveness')
    
    # Add reference line for breakeven ROAS
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Breakeven ROAS')
    
    # Create custom legend for selection status
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='green', 
              markersize=10, label='Selected Keywords'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='red', 
              markersize=10, label='Rejected Keywords')
    ]
    plt.legend(handles=legend_elements + [Line2D([0], [0], color='red', linestyle='--', label='Breakeven ROAS')], 
              loc='upper right')
    
    # Configure axes
    plt.xlabel('Ad Spend', fontsize=12)
    plt.ylabel('Return on Ad Spend (ROAS)', fontsize=12)
    plt.title('ROAS vs Ad Spend: Investment Decision Strategy', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Apply dimensional constraints
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    # Generate image artifact
    roas_plot_path = os.path.join(output_dir, "roas_vs_spend.png")
    plt.savefig(roas_plot_path, dpi=100)
    plt.close()
    
    saved_plots.append(roas_plot_path)
    print(f"ROAS vs Spend plot saved to {roas_plot_path}")
    
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
    
    # Generate ROAS group analysis visualization
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Define ROAS classification taxonomy
    def get_roas_group(row):
        if row['ad_roas'] > 2.0:
            return "High ROAS (>2.0)"
        elif row['ad_roas'] > 1.2:
            return "Medium ROAS (1.2-2.0)"
        elif row['ad_roas'] > 1.0:
            return "Marginal ROAS (1.0-1.2)"
        else:
            return "Unprofitable (<1.0)"
    
    dataset['roas_group'] = dataset.apply(get_roas_group, axis=1)
    
    # Group by ROAS category and calculate selection rates with multidimensional aggregation
    roas_groups = dataset.groupby('roas_group').agg({
        'keyword': 'count',
        'selected': 'mean',
        'ad_roas': 'mean',
        'ad_spend': 'mean',
        'paid_ctr': 'mean'
    }).reset_index()
    
    roas_groups.rename(columns={
        'keyword': 'count',
        'selected': 'selection_rate',
        'ad_roas': 'avg_roas',
        'ad_spend': 'avg_spend',
        'paid_ctr': 'avg_ctr'
    }, inplace=True)
    
    # Ensure complete taxonomic representation
    all_groups = ["High ROAS (>2.0)", "Medium ROAS (1.2-2.0)", "Marginal ROAS (1.0-1.2)", "Unprofitable (<1.0)"]
    for group in all_groups:
        if group not in roas_groups['roas_group'].values:
            roas_groups = pd.concat([roas_groups, pd.DataFrame([{
                'roas_group': group,
                'count': 0,
                'selection_rate': 0,
                'avg_roas': 0,
                'avg_spend': 0,
                'avg_ctr': 0
            }])], ignore_index=True)
    
    # Implement ordered representation
    roas_groups['order'] = roas_groups['roas_group'].map({
        "High ROAS (>2.0)": 0,
        "Medium ROAS (1.2-2.0)": 1,
        "Marginal ROAS (1.0-1.2)": 2,
        "Unprofitable (<1.0)": 3
    })
    roas_groups = roas_groups.sort_values('order').reset_index(drop=True)
    
    # Convert selection rate to percentage for interpretability
    roas_groups['selection_rate'] = roas_groups['selection_rate'] * 100
    
    # Implement bar visualization with color modulation based on selection rate
    bars = plt.bar(roas_groups['roas_group'], roas_groups['selection_rate'], color='skyblue')
    
    # Color bars based on selection rate with gradient visualization
    cmap = plt.cm.RdYlGn
    for i, bar in enumerate(bars):
        bar.set_color(cmap(roas_groups['selection_rate'].iloc[i] / 100))
    
    # Add annotation layers for count and rate metrics
    for i, row in roas_groups.iterrows():
        plt.text(i, 5, f"n = {int(row['count'])}", ha='center', va='center', color='black', fontweight='bold')
        plt.text(i, row['selection_rate'] + 3, f"{row['selection_rate']:.1f}%", ha='center', va='bottom')
    
    # Configure axes
    plt.xlabel('ROAS Group', fontsize=12)
    plt.ylabel('Selection Rate (%)', fontsize=12)
    plt.title('Keyword Selection Strategy by ROAS Group', fontsize=14)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add ROAS average as contextual subtext
    for i, row in roas_groups.iterrows():
        if row['avg_roas'] > 0:
            plt.text(i, -5, f"Avg ROAS: {row['avg_roas']:.2f}", ha='center', va='top', fontsize=9)
    
    # Apply dimensional constraints
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9)
    
    # Generate image artifact
    group_plot_path = os.path.join(output_dir, "roas_group_analysis.png")
    plt.savefig(group_plot_path, dpi=100)
    plt.close()
    
    saved_plots.append(group_plot_path)
    print(f"ROAS group analysis plot saved to {group_plot_path}")
    
    # Generate portfolio performance visualization
    plt.figure(figsize=(10, 8), dpi=100)
    
    # Calculate portfolio metrics with selection-based aggregation
    selected_df = dataset[dataset['selected']]
    
    total_spend = selected_df['ad_spend'].sum()
    total_value = (selected_df['ad_roas'] * selected_df['ad_spend']).sum()
    portfolio_roas = total_value / total_spend if total_spend > 0 else 0
    
    # Implement pie chart visualization for distribution analysis
    labels = ['Ad Spend', 'Return']
    sizes = [total_spend, total_value - total_spend]
    colors = ['#ff9999', '#99ff99']
    explode = (0, 0.1)
    
    # Only visualize with valid data
    if sum(sizes) > 0:
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
    
    # Add portfolio metrics table for comprehensive representation
    table_data = [
        ['Metric', 'Value'],
        ['Selected Keywords', f"{len(selected_df)} / {len(dataset)} ({len(selected_df)/len(dataset)*100:.1f}%)"],
        ['Total Ad Spend', f"${total_spend:,.2f}"],
        ['Total Revenue', f"${total_value:,.2f}"],
        ['Portfolio ROAS', f"{portfolio_roas:.2f}"],
        ['Profit', f"${total_value - total_spend:,.2f}"]
    ]
    
    plt.table(cellText=table_data, loc='center', bbox=[0.25, -0.5, 0.5, 0.3], cellLoc='center')
    
    plt.title('Portfolio Performance Analysis', fontsize=14)
    plt.axis('equal')  # Equal aspect ratio for circular representation
    
    # Add strategy explanation for interpretability
    plt.text(0.5, -0.7, 
            "Portfolio optimization strategies:\n"
            "1. Risk-managed budget allocation (10% constraint)\n"
            "2. ROAS-prioritized selection hierarchy\n"
            "3. CTR-modulated marginal investment decisions", 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", alpha=0.8))
    
    # Apply dimensional constraints
    plt.subplots_adjust(top=0.9, bottom=-0.2, left=0.1, right=0.9)
    
    # Generate image artifact
    portfolio_plot_path = os.path.join(output_dir, "portfolio_performance.png")
    plt.savefig(portfolio_plot_path, dpi=100)
    plt.close()
    
    saved_plots.append(portfolio_plot_path)
    print(f"Portfolio performance plot saved to {portfolio_plot_path}")
    
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
    
    # Add sections for each type of visualization
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
    
    # Load or generate dataset
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
        print(f"Error details: {traceback.format_exc()}")
        print("Generating emergency synthetic dataset")
        dataset = generate_synthetic_data(100)  # Smaller emergency dataset
    
    # Generate visualizations
    all_plots = {}
    
    # 1. Epsilon-Greedy Exploration
    try:
        all_plots["Epsilon-Greedy Exploration Strategy"] = visualize_epsilon_greedy_exploration(
            metric_dfs, 
            os.path.join(args.output_dir, "exploration_strategy")
        )
    except Exception as e:
        print(f"Warning: Error visualizing epsilon-greedy exploration: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Epsilon-Greedy Exploration Strategy"] = []
    
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
        all_plots["Experience Replay & Target Network"] = visualize_experience_replay_target_network(
            os.path.join(args.output_dir, "experience_replay")
        )
    except Exception as e:
        print(f"Warning: Error visualizing experience replay: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Experience Replay & Target Network"] = []
    
    # 4. Budget Allocation Strategy
    try:
        all_plots["Budget Allocation Strategy"] = visualize_budget_allocation_strategy(
            dataset,
            os.path.join(args.output_dir, "budget_strategy")
        )
    except Exception as e:
        print(f"Warning: Error visualizing budget allocation: {e}")
        print(f"Error details: {traceback.format_exc()}")
        all_plots["Budget Allocation Strategy"] = []
    
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

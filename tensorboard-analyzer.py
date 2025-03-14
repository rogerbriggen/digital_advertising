#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import glob
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def find_event_files(log_dir):
    """Find all TensorBoard event files in the given directory and subdirectories."""
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    return event_files


def extract_metrics_from_events(event_files):
    """Extract metrics from TensorBoard event files."""
    metrics = defaultdict(list)
    
    for event_file in event_files:
        print(f"Processing {event_file}")
        try:
            event_acc = EventAccumulator(event_file)
            event_acc.Reload()
            
            # Extract scalar metrics
            for tag in event_acc.Tags()['scalars']:
                for event in event_acc.Scalars(tag):
                    metrics[tag].append({
                        'step': event.step,
                        'time': event.wall_time,
                        'value': event.value
                    })
            
            # Try to extract text metrics
            for tag in event_acc.Tags().get('tensors', []):
                try:
                    for event in event_acc.Tensors(tag):
                        metrics[tag].append({
                            'step': event.step,
                            'time': event.wall_time,
                            'value': str(event.tensor_proto)
                        })
                except Exception as e:
                    print(f"Error extracting tensor {tag}: {e}")
                    
        except Exception as e:
            print(f"Error processing {event_file}: {e}")
    
    # Convert to DataFrames
    result = {}
    for tag, events in metrics.items():
        if events:
            result[tag] = pd.DataFrame(events)
    
    return result


def analyze_training_progress(metrics, output_dir):
    """Analyze and visualize training progress metrics."""
    os.makedirs(output_dir, exist_ok=True)
    plots = []
    
    # Find loss and performance metrics
    loss_metric = next((m for m in metrics.keys() if 'loss' in m.lower()), None)
    performance_metric = next((m for m in metrics.keys() if 'performance' in m.lower() or 'reward' in m.lower()), None)
    
    if loss_metric:
        loss_df = metrics[loss_metric]
        
        # Smooth the loss curve for better visualization
        if len(loss_df) > 1:
            window_size = min(10, max(1, len(loss_df) // 10))  # Adaptive window size
            loss_df['smoothed_value'] = loss_df['value'].rolling(window=window_size, min_periods=1).mean()
        else:
            loss_df['smoothed_value'] = loss_df['value']
        
        plt.figure(figsize=(12, 6))
        plt.plot(loss_df['step'], loss_df['value'], alpha=0.3, color='blue', label='Raw Loss')
        plt.plot(loss_df['step'], loss_df['smoothed_value'], linewidth=2, color='darkblue', label='Smoothed Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add trend line
        if len(loss_df) > 10:
            try:
                z = np.polyfit(loss_df['step'], loss_df['smoothed_value'], 1)
                p = np.poly1d(z)
                plt.plot(loss_df['step'], p(loss_df['step']), "r--", alpha=0.8, label='Trend Line')
                plt.legend()
            except Exception as e:
                print(f"Warning: Could not create trend line: {e}")
        
        loss_plot_path = os.path.join(output_dir, "loss_analysis.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots.append(loss_plot_path)
    
    if performance_metric:
        perf_df = metrics[performance_metric]
        
        # Calculate improvement over time
        if len(perf_df) > 1:
            try:
                perf_df['improvement'] = perf_df['value'].pct_change() * 100
                
                # Plot performance and improvement
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                
                # Performance plot
                ax1.plot(perf_df['step'], perf_df['value'], 'g-', linewidth=2)
                ax1.set_title('Agent Performance Over Training')
                ax1.set_ylabel('Reward')
                ax1.grid(True, alpha=0.3)
                
                # Highlight best performance
                best_idx = perf_df['value'].idxmax()
                best_step = perf_df.loc[best_idx, 'step']
                best_value = perf_df.loc[best_idx, 'value']
                ax1.scatter([best_step], [best_value], color='red', s=100, zorder=5)
                ax1.annotate(f'Best: {best_value:.2f}', 
                             xy=(best_step, best_value),
                             xytext=(10, 10),
                             textcoords='offset points',
                             arrowprops=dict(arrowstyle='->'))
                
                # Improvement plot
                ax2.bar(perf_df['step'][1:], perf_df['improvement'][1:], color='blue', alpha=0.6)
                ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                ax2.set_title('Percentage Improvement Between Evaluations')
                ax2.set_xlabel('Training Steps')
                ax2.set_ylabel('% Change')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                perf_plot_path = os.path.join(output_dir, "performance_analysis.png")
                plt.savefig(perf_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                plots.append(perf_plot_path)
            except Exception as e:
                print(f"Warning: Could not create performance improvement plot: {e}")
                
                # Fallback to simple performance plot
                plt.figure(figsize=(12, 6))
                plt.plot(perf_df['step'], perf_df['value'], 'g-', linewidth=2)
                plt.title('Agent Performance Over Training')
                plt.xlabel('Training Steps')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
                
                if len(perf_df) > 0:
                    best_idx = perf_df['value'].idxmax()
                    best_step = perf_df.loc[best_idx, 'step']
                    best_value = perf_df.loc[best_idx, 'value']
                    plt.scatter([best_step], [best_value], color='red', s=100, zorder=5)
                    plt.annotate(f'Best: {best_value:.2f}', 
                                xy=(best_step, best_value),
                                xytext=(10, 10),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='->'))
                
                perf_plot_path = os.path.join(output_dir, "performance_simple.png")
                plt.savefig(perf_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                plots.append(perf_plot_path)
    
    # Combined analysis if both metrics exist
    if loss_metric and performance_metric:
        loss_df = metrics[loss_metric]
        perf_df = metrics[performance_metric]
        
        try:
            # Use a simpler approach to compare trends
            plt.figure(figsize=(12, 6))
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color1 = 'tab:red'
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss Value', color=color1)
            ax1.plot(loss_df['step'], loss_df['value'], color=color1, alpha=0.7, label='Loss')
            ax1.tick_params(axis='y', labelcolor=color1)
            
            ax2 = ax1.twinx()
            color2 = 'tab:blue'
            ax2.set_ylabel('Performance', color=color2)
            ax2.plot(perf_df['step'], perf_df['value'], color=color2, alpha=0.7, label='Performance')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            plt.title('Loss and Performance Trends')
            
            # Add legend for both lines
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
            
            plt.grid(True, alpha=0.3)
            
            combined_plot_path = os.path.join(output_dir, "loss_performance_trends.png")
            plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots.append(combined_plot_path)
        except Exception as e:
            print(f"Warning: Could not create combined analysis plot: {e}")
    
    return plots


def analyze_learning_stability(metrics, output_dir):
    """Analyze learning stability based on metrics."""
    os.makedirs(output_dir, exist_ok=True)
    plots = []
    
    loss_metric = next((m for m in metrics.keys() if 'loss' in m.lower()), None)
    
    if loss_metric:
        loss_df = metrics[loss_metric]
        
        if len(loss_df) < 10:
            print("Warning: Not enough loss data points for stability analysis")
            return plots
            
        try:
            # Calculate rolling statistics
            window_size = max(10, len(loss_df) // 20)  # adaptive window size
            loss_df['rolling_mean'] = loss_df['value'].rolling(window=window_size, min_periods=1).mean()
            loss_df['rolling_std'] = loss_df['value'].rolling(window=window_size, min_periods=1).std()
            loss_df['upper_bound'] = loss_df['rolling_mean'] + 2 * loss_df['rolling_std']
            loss_df['lower_bound'] = loss_df['rolling_mean'] - 2 * loss_df['rolling_std']
            
            # Stability visualization
            plt.figure(figsize=(12, 6))
            plt.plot(loss_df['step'], loss_df['value'], 'b-', alpha=0.5, label='Loss')
            plt.plot(loss_df['step'], loss_df['rolling_mean'], 'r-', label=f'Rolling Mean (window={window_size})')
            plt.fill_between(loss_df['step'], 
                             loss_df['lower_bound'], 
                             loss_df['upper_bound'], 
                             color='red', 
                             alpha=0.2, 
                             label='±2 Std Dev')
            
            plt.title('Learning Stability Analysis')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Calculate coefficient of variation for stability
            with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by zero warnings
                cv = loss_df['rolling_std'] / loss_df['rolling_mean']
                cv = cv.replace([np.inf, -np.inf], np.nan)  # Replace infinity with NaN
                avg_cv = cv.mean()  # NaNs are ignored in mean calculation
            
            if not np.isnan(avg_cv):
                plt.annotate(f'Avg. Coefficient of Variation: {avg_cv:.4f}', 
                             xy=(0.05, 0.05), 
                             xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            
            stability_plot_path = os.path.join(output_dir, "learning_stability.png")
            plt.savefig(stability_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots.append(stability_plot_path)
            
            # Detect and visualize convergence
            if len(loss_df) > window_size * 2:
                try:
                    # Calculate rate of change
                    loss_df['delta'] = loss_df['rolling_mean'].diff().abs()
                    # Handle division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        loss_df['pct_change'] = loss_df['delta'] / loss_df['rolling_mean'].shift(1) * 100
                        loss_df['pct_change'] = loss_df['pct_change'].replace([np.inf, -np.inf], np.nan)
                    
                    # Fill NaN values with a high percentage to avoid false convergence detection
                    loss_df['pct_change'] = loss_df['pct_change'].fillna(100)
                    
                    # Convergence threshold (arbitrary - can be adjusted)
                    convergence_threshold = 1.0  # 1% change
                    
                    # Find where convergence occurs
                    convergence_points = loss_df[loss_df['pct_change'] < convergence_threshold].copy()
                    
                    if not convergence_points.empty:
                        # Find the first stable convergence (where we have at least window_size consecutive points)
                        stable_points = []
                        current_run = []
                        
                        for idx, row in convergence_points.iterrows():
                            if not current_run or idx == current_run[-1] + 1:
                                current_run.append(idx)
                            else:
                                if len(current_run) >= window_size:
                                    stable_points.extend(current_run)
                                current_run = [idx]
                        
                        if len(current_run) >= window_size:
                            stable_points.extend(current_run)
                        
                        # Visualize convergence
                        plt.figure(figsize=(12, 6))
                        plt.plot(loss_df['step'], loss_df['pct_change'], 'g-', alpha=0.7)
                        plt.axhline(y=convergence_threshold, color='r', linestyle='--', label=f'Threshold ({convergence_threshold}%)')
                        
                        if stable_points:
                            convergence_step = loss_df.loc[stable_points[0], 'step']
                            plt.axvline(x=convergence_step, color='b', linestyle='--', 
                                        label=f'Convergence at step {convergence_step}')
                        
                        plt.title('Convergence Analysis')
                        plt.xlabel('Training Steps')
                        plt.ylabel('Percent Change in Loss (rolling window)')
                        plt.yscale('log')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        convergence_plot_path = os.path.join(output_dir, "convergence_analysis.png")
                        plt.savefig(convergence_plot_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        plots.append(convergence_plot_path)
                except Exception as e:
                    print(f"Warning: Error in convergence analysis: {e}")
        except Exception as e:
            print(f"Warning: Error in stability analysis: {e}")
    
    return plots


def create_training_report(metrics, all_plots, output_dir):
    """Create an HTML report summarizing the training analysis."""
    report_path = os.path.join(output_dir, "training_analysis_report.html")
    
    # Extract key metrics
    performance_metric = next((m for m in metrics.keys() if 'performance' in m.lower() or 'reward' in m.lower()), None)
    
    best_performance = "N/A"
    if performance_metric and not metrics[performance_metric].empty:
        best_performance = f"{metrics[performance_metric]['value'].max():.2f}"
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RL Training Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 30px;
            }}
            .summary {{
                background-color: #f8f9fa;
                border-left: 4px solid #2980b9;
                padding: 15px;
                margin: 20px 0;
            }}
            .metrics {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .metric {{
                background-color: #f0f0f0;
                border-radius: 5px;
                padding: 15px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
                margin: 10px 0;
            }}
            .metric-label {{
                color: #7f8c8d;
                font-size: 14px;
            }}
            .plot {{
                margin: 30px 0;
                text-align: center;
            }}
            .plot img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .caption {{
                margin-top: 10px;
                font-style: italic;
                color: #7f8c8d;
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 12px;
                color: #7f8c8d;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Reinforcement Learning Training Analysis</h1>
        
        <div class="summary">
            <p>This report analyzes the training progress of the reinforcement learning agent for digital advertising optimization.</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Best Performance</div>
                <div class="metric-value">{best_performance}</div>
                <div class="metric-label">Maximum reward achieved</div>
            </div>
        </div>
    """
    
    # Add sections for each category of plots
    for section, plots in all_plots.items():
        if plots:
            html += f"""
            <h2>{section}</h2>
            """
            
            for plot_path in plots:
                plot_name = os.path.basename(plot_path)
                plot_rel_path = os.path.relpath(plot_path, output_dir)
                caption = plot_name.replace('.png', '').replace('_', ' ').title()
                
                html += f"""
                <div class="plot">
                    <img src="{plot_rel_path}" alt="{caption}">
                    <div class="caption">{caption}</div>
                </div>
                """
    
    html += f"""
        <footer>
            <p>Generated on {timestamp}</p>
        </footer>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze TensorBoard logs in detail")
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--output_dir", type=str, default="training_analysis", help="Output directory for visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all TensorBoard event files
    try:
        event_files = find_event_files(args.logdir)
        
        if not event_files:
            print(f"No TensorBoard event files found in {args.logdir}")
            return
        
        print(f"Found {len(event_files)} TensorBoard event files")
        
        # Extract metrics from event files
        metrics = extract_metrics_from_events(event_files)
        
        if not metrics:
            print("No metrics found in the TensorBoard logs")
            return
        
        print(f"Extracted metrics: {list(metrics.keys())}")
        
        # Perform analyses
        all_plots = {}
        
        # Training progress analysis
        try:
            progress_plots = analyze_training_progress(metrics, os.path.join(args.output_dir, "progress"))
            if progress_plots:
                all_plots["Training Progress"] = progress_plots
        except Exception as e:
            print(f"Error in training progress analysis: {e}")
        
        # Learning stability analysis
        try:
            stability_plots = analyze_learning_stability(metrics, os.path.join(args.output_dir, "stability"))
            if stability_plots:
                all_plots["Learning Stability"] = stability_plots
        except Exception as e:
            print(f"Error in learning stability analysis: {e}")
        
        # Create report
        if all_plots:
            try:
                report_path = create_training_report(metrics, all_plots, args.output_dir)
                print(f"Analysis complete! Report available at: {report_path}")
            except Exception as e:
                print(f"Error creating training report: {e}")
                print("Individual plots are still available in the output directory.")
        else:
            print("No plots were generated from the metrics")
    
    except Exception as e:
        print(f"Unexpected error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

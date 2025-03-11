#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import deque
import time

# TorchRL imports
from torchrl.envs import EnvBase
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from typing import Optional
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.data import OneHot, Bounded, Unbounded, Binary, MultiCategorical, Composite
from torchrl.data import LazyTensorStorage, ReplayBuffer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

# Generate Realistic Synthetic Data with improved correlations from Roger's script
def generate_synthetic_data(num_samples=1000):
    """Generate synthetic advertising data with realistic correlations."""
    base_difficulty = np.random.beta(2.5, 3.5, num_samples)
    data = {
        "keyword": [f"Keyword_{i}" for i in range(num_samples)],
        "competitiveness": np.random.beta(2, 3, num_samples),
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
    
    # Add realistic correlations
    data["difficulty_score"] = 0.7 * data["competitiveness"] + 0.3 * base_difficulty
    data["organic_rank"] = 1 + np.floor(9 * data["difficulty_score"] + np.random.normal(0, 1, num_samples).clip(-2, 2))
    data["organic_rank"] = data["organic_rank"].clip(1, 10).astype(int)
    
    # CTR follows a realistic distribution and correlates negatively with rank
    base_ctr = np.random.beta(1.5, 10, num_samples)
    rank_effect = (11 - data["organic_rank"]) / 10
    data["organic_ctr"] = (base_ctr * rank_effect * 0.3).clip(0.01, 0.3)
    
    # Organic clicks based on CTR and a base impression count
    base_impressions = np.random.lognormal(8, 1, num_samples).astype(int)
    data["organic_clicks"] = (base_impressions * data["organic_ctr"]).astype(int)
    
    # Paid CTR correlates with organic CTR but with more variance
    data["paid_ctr"] = (data["organic_ctr"] * np.random.normal(1, 0.3, num_samples)).clip(0.01, 0.25)
    
    # Paid clicks
    paid_impressions = np.random.lognormal(7, 1.2, num_samples).astype(int)
    data["paid_clicks"] = (paid_impressions * data["paid_ctr"]).astype(int)
    
    # Cost per click higher for more competitive keywords
    data["cost_per_click"] = (0.5 + 9.5 * data["competitiveness"] * np.random.normal(1, 0.2, num_samples)).clip(0.1, 10)
    
    # Ad spend based on CPC and clicks
    data["ad_spend"] = data["paid_clicks"] * data["cost_per_click"]
    
    # Conversion rate with realistic e-commerce distribution
    data["conversion_rate"] = np.random.beta(1.2, 15, num_samples).clip(0.01, 0.3)
    
    # Ad conversions
    data["ad_conversions"] = (data["paid_clicks"] * data["conversion_rate"]).astype(int)
    
    # Conversion value with variance
    base_value = np.random.lognormal(4, 1, num_samples)
    data["conversion_value"] = data["ad_conversions"] * base_value
    
    # Cost per acquisition
    with np.errstate(divide='ignore', invalid='ignore'):
        data["cost_per_acquisition"] = np.where(
            data["ad_conversions"] > 0, 
            data["ad_spend"] / data["ad_conversions"], 
            500  # Default high CPA for no conversions
        ).clip(5, 500)
    
    # ROAS (Return on Ad Spend)
    with np.errstate(divide='ignore', invalid='ignore'):
        data["ad_roas"] = np.where(
            data["ad_spend"] > 0,
            data["conversion_value"] / data["ad_spend"],
            0
        ).clip(0.5, 5)
    
    # Impression share (competitive keywords have lower share)
    data["impression_share"] = (1 - 0.6 * data["competitiveness"] * np.random.normal(1, 0.2, num_samples)).clip(0.1, 1.0)
    
    return pd.DataFrame(data)

# Define feature columns to use in the environment
feature_columns = [
    "competitiveness", 
    "difficulty_score", 
    "organic_rank", 
    "organic_clicks", 
    "organic_ctr", 
    "paid_clicks", 
    "paid_ctr", 
    "ad_spend", 
    "ad_conversions", 
    "ad_roas", 
    "conversion_rate", 
    "cost_per_click"
]

# Utility function to get real-world keywords
def get_keywords():
    return [
        "investments", "stocks", "crypto", "cryptocurrency", "bitcoin", 
        "real estate", "gold", "bonds", "broker", "finance", "trading", 
        "forex", "etf", "investment fund", "investment strategy", 
        "investment advice", "investment portfolio", "investment opportunities", 
        "investment options", "investment calculator", "investment plan", 
        "investment account", "investment return", "investment risk", 
        "investment income", "investment growth", "investment loss", 
        "investment profit", "investment return calculator", 
        "investment return formula", "investment return rate"
    ]

# Function to get a batch of data for multiple keywords
def get_entry_from_dataset(df, index, keyword_count=None):
    if keyword_count is None:
        # Count unique keywords on first call
        if not hasattr(get_entry_from_dataset, "unique_keywords"):
            seen_keywords = set()
            for i, row in df.iterrows():
                keyword = row['keyword']
                if keyword in seen_keywords:
                    break
                seen_keywords.add(keyword)
            get_entry_from_dataset.unique_keywords = seen_keywords
            get_entry_from_dataset.keywords_amount = len(seen_keywords)
        
        keyword_count = get_entry_from_dataset.keywords_amount
    
    start_idx = index * keyword_count
    end_idx = start_idx + keyword_count
    
    # Make sure we don't go out of bounds
    if end_idx > len(df):
        return None
        
    return df.iloc[start_idx:end_idx].reset_index(drop=True)

# TorchRL Environment Implementation
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset, initial_cash=100000.0, device="cpu"):
        super().__init__(device=device)
        self.initial_cash = initial_cash
        self.dataset = dataset
        self.num_features = len(feature_columns)
        
        # Get the first entry to determine the number of keywords
        first_entry = get_entry_from_dataset(self.dataset, 0)
        self.num_keywords = first_entry.shape[0]
        
        # Define the action spec (OneHot: select one keyword or none)
        self.action_spec = OneHot(n=self.num_keywords + 1)  # +1 for the "buy nothing" action
        
        # Define the reward spec
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        
        # Define the observation spec
        self.observation_spec = Composite(
            observation = Composite(
                keyword_features=Unbounded(shape=(self.num_keywords, self.num_features), dtype=torch.float32),
                cash=Unbounded(shape=(1,), dtype=torch.float32),
                holdings=Bounded(low=0, high=1, shape=(self.num_keywords,), dtype=torch.int, domain="discrete")
            ),
            step_count=Unbounded(shape=(1,), dtype=torch.int64)
        )
        
        # Define the done spec
        self.done_spec = Composite(
            done=Binary(shape=(1,), dtype=torch.bool),
            terminated=Binary(shape=(1,), dtype=torch.bool),
            truncated=Binary(shape=(1,), dtype=torch.bool)
        )
        
        self.reset()

    def _reset(self, tensordict=None):
        self.current_step = 0
        self.holdings = torch.zeros(self.num_keywords, dtype=torch.int, device=self.device)
        self.cash = self.initial_cash
        
        # Get feature data for the first step
        current_data = get_entry_from_dataset(self.dataset, self.current_step)
        keyword_features = torch.tensor(current_data[feature_columns].values, dtype=torch.float32, device=self.device)
        
        # Create the initial observation
        obs = TensorDict({
            "keyword_features": keyword_features,
            "cash": torch.tensor([self.cash], dtype=torch.float32, device=self.device),
            "holdings": self.holdings.clone()
        }, batch_size=[])
        
        # Initialize or update tensordict
        if tensordict is None:
            tensordict = TensorDict({
                "done": torch.tensor(False, dtype=torch.bool, device=self.device),
                "observation": obs,
                "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
                "terminated": torch.tensor(False, dtype=torch.bool, device=self.device),
                "truncated": torch.tensor(False, dtype=torch.bool, device=self.device)
            }, batch_size=[])
        else:
            tensordict.update({
                "done": torch.tensor(False, dtype=torch.bool, device=self.device),
                "observation": obs,
                "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
                "terminated": torch.tensor(False, dtype=torch.bool, device=self.device),
                "truncated": torch.tensor(False, dtype=torch.bool, device=self.device)
            })
        
        self.obs = obs
        return tensordict

    def _step(self, tensordict):
        # Get the action from the input tensor dictionary
        action = tensordict["action"]
        
        # Find which action was selected (which position has 1)
        true_indices = torch.nonzero(action, as_tuple=True)[0]
        action_idx = true_indices[0].item() if len(true_indices) > 0 else self.num_keywords
        
        # Get current data
        current_data = get_entry_from_dataset(self.dataset, self.current_step)
        
        # Update holdings based on action (only one keyword is selected)
        new_holdings = torch.zeros_like(self.holdings)
        if action_idx < self.num_keywords:
            new_holdings[action_idx] = 1
        self.holdings = new_holdings
        
        # Calculate reward
        reward = self._compute_reward(action, current_data)
        
        # Move to next step
        self.current_step += 1
        
        # Check if we've reached the end of the dataset
        max_steps = len(self.dataset) // self.num_keywords - 1
        terminated = self.current_step >= max_steps
        truncated = False
        
        # Get next state data if not terminated
        if not terminated:
            next_data = get_entry_from_dataset(self.dataset, self.current_step)
            next_keyword_features = torch.tensor(next_data[feature_columns].values, dtype=torch.float32, device=self.device)
        else:
            # If terminated, use current data (doesn't matter as episode is ending)
            next_keyword_features = torch.tensor(current_data[feature_columns].values, dtype=torch.float32, device=self.device)
        
        # Create next observation
        next_obs = TensorDict({
            "keyword_features": next_keyword_features,
            "cash": torch.tensor([self.cash], dtype=torch.float32, device=self.device),
            "holdings": self.holdings.clone()
        }, batch_size=[])
        
        # Update state
        self.obs = next_obs
        
        # Create result TensorDict
        next_td = TensorDict({
            "done": torch.tensor(terminated or truncated, dtype=torch.bool, device=self.device),
            "observation": next_obs,
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "step_count": torch.tensor([self.current_step], dtype=torch.int64, device=self.device),
            "terminated": torch.tensor(terminated, dtype=torch.bool, device=self.device),
            "truncated": torch.tensor(truncated, dtype=torch.bool, device=self.device)
        }, batch_size=tensordict.batch_size)
        
        return next_td

    def _compute_reward(self, action, current_data):
        """Compute reward based on the selected keyword's metrics"""
        if action.sum() == 0 or torch.argmax(action.float()).item() == self.num_keywords:
            # "Do nothing" action
            return 0.0
        
        selected_idx = torch.argmax(action.float()).item()
        selected_keyword = current_data.iloc[selected_idx]
        
        # Calculate reward based on ROAS and CTR
        roas = selected_keyword["ad_roas"]
        ctr = selected_keyword["paid_ctr"]
        ad_spend = selected_keyword["ad_spend"]
        
        # Positive reward for high ROAS keywords with significant ad spend
        if roas > 2.0 and ad_spend > 1000:
            reward = 2.0
        # Medium reward for profitable keywords
        elif roas > 1.0:
            reward = 1.0
        # Medium reward for high CTR keywords
        elif ctr > 0.15:
            reward = 1.0
        # Slight penalty for unprofitable choices
        else:
            reward = -0.5
            
        return reward
        
    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

# Module to flatten and combine inputs for the neural network
class FlattenInputs(nn.Module):
    def forward(self, keyword_features, cash, holdings):
        # Check if we have a batch dimension
        has_batch = keyword_features.dim() > 2
        
        if has_batch:
            batch_size = keyword_features.shape[0]
            # Flatten keyword features while preserving batch dimension: 
            # [batch, num_keywords, feature_dim] -> [batch, num_keywords * feature_dim]
            flattened_features = keyword_features.reshape(batch_size, -1)
            
            # Ensure cash has correct dimensions [batch, 1]
            if cash.dim() == 1:  # [batch]
                cash = cash.unsqueeze(-1)  # [batch, 1]
            elif cash.dim() == 0:  # scalar
                cash = cash.unsqueeze(0).expand(batch_size, 1)  # [batch, 1]
            
            # Ensure holdings has correct dimensions [batch, num_keywords]
            if holdings.dim() == 1:  # [num_keywords]
                holdings = holdings.unsqueeze(0).expand(batch_size, -1)  # [batch, num_keywords]
            
            # Convert holdings to float
            holdings = holdings.float()
            
            # Combine all inputs along dimension 1
            combined = torch.cat([flattened_features, cash, holdings], dim=1)
        else:
            # No batch dimension - single sample case
            # Flatten keyword features: [num_keywords, feature_dim] -> [num_keywords * feature_dim]
            flattened_features = keyword_features.reshape(-1)
            
            # Ensure cash has a dimension
            cash = cash.unsqueeze(-1) if cash.dim() == 0 else cash
            
            # Convert holdings to float
            holdings = holdings.float()
            
            # Combine all inputs
            combined = torch.cat([flattened_features, cash, holdings], dim=0)
            
        return combined

# Visualization functions
# Fix for the visualize_training_progress function
def visualize_training_progress(metrics, output_dir="plots", window_size=20):
    """Visualize training metrics including rewards, losses, and exploration rate."""
    os.makedirs(output_dir, exist_ok=True)
    
    rewards = metrics["rewards"]
    losses = metrics["losses"]
    epsilons = metrics["epsilon_values"]
    
    # Ensure tensors are converted to CPU NumPy arrays
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
    if isinstance(losses, torch.Tensor):
        losses = losses.cpu().numpy()
    if isinstance(epsilons, torch.Tensor):
        epsilons = epsilons.cpu().numpy()
    
    if not rewards:
        print("No rewards to visualize")
        return None
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("RL Training Progress", fontsize=16)
    
    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, color='blue', label="Episode Rewards")
    
    if len(rewards) >= window_size:
        # Add smoothed rewards line
        smoothed_rewards = []
        for i in range(len(rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(rewards[i:i+window_size]))
        axes[0].plot(range(window_size-1, len(rewards)), smoothed_rewards, 
                   color='red', linewidth=2, label=f"Moving Average ({window_size})")
    
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Training Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot losses
    if losses:
        axes[1].plot(losses, color='purple', alpha=0.5, label="Training Loss")
        
        if len(losses) >= window_size:
            # Add smoothed losses line
            smoothed_losses = []
            for i in range(len(losses) - window_size + 1):
                smoothed_losses.append(np.mean(losses[i:i+window_size]))
            axes[1].plot(range(window_size-1, len(losses)), smoothed_losses, 
                       color='darkred', linewidth=2, label=f"Moving Average ({window_size})")
        
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot exploration rate
    if epsilons:
        axes[2].plot(epsilons, color='green', label="Exploration Rate (ε)")
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("Episodes")
        axes[2].set_ylabel("Epsilon (ε)")
        axes[2].set_title("Exploration Rate")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/training_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path

def visualize_evaluation(eval_results, feature_columns, output_dir="plots"):
    """Visualize the evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert any tensors to numpy arrays
    action_counts = eval_results["action_counts"]
    rewards = eval_results["rewards"]
    feature_importance = eval_results["feature_importance"]
    
    # Check and convert tensors
    if isinstance(action_counts, torch.Tensor):
        action_counts = action_counts.cpu().numpy()
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
    if isinstance(feature_importance, torch.Tensor):
        feature_importance = feature_importance.cpu().numpy()
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Ad Optimization Evaluation Results", fontsize=16)
    
    # 1. Action Distribution (Top Left)
    action_labels = ["No Action"] + [f"Keyword {i}" for i in range(len(action_counts)-1)]
    axs[0, 0].bar(action_labels, action_counts)
    axs[0, 0].set_title("Action Distribution")
    axs[0, 0].set_ylabel("Count")
    plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha="right")
    
    # 2. Rewards Distribution (Top Right)
    axs[0, 1].hist(rewards, bins=20, alpha=0.7, color='blue')
    axs[0, 1].axvline(x=np.mean(rewards), color='r', linestyle='--', 
                     label=f'Mean: {np.mean(rewards):.2f}')
    axs[0, 1].set_title("Reward Distribution")
    axs[0, 1].set_xlabel("Reward")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].legend()
    
    # 3. Feature Importance (Bottom Left)
    if feature_importance is not None:
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)
        axs[1, 0].barh([feature_columns[i] for i in sorted_idx], 
                      [feature_importance[i] for i in sorted_idx])
        axs[1, 0].set_title("Feature Importance")
        axs[1, 0].set_xlabel("Importance Score")
    else:
        axs[1, 0].text(0.5, 0.5, "Feature importance not available", 
                      ha='center', va='center', fontsize=12)
    
    # 4. Performance Over Time (Bottom Right)
    if len(rewards) > 0:
        # Calculate cumulative average reward
        cum_rewards = np.cumsum(rewards)
        cum_avg = cum_rewards / np.arange(1, len(cum_rewards) + 1)
        
        axs[1, 1].plot(cum_avg, color='green', label='Cumulative Average Reward')
        axs[1, 1].set_title("Performance Over Time")
        axs[1, 1].set_xlabel("Steps")
        axs[1, 1].set_ylabel("Cumulative Average Reward")
        axs[1, 1].legend()
    else:
        axs[1, 1].text(0.5, 0.5, "No reward data available", 
                      ha='center', va='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/evaluation_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path

# Training function that works with TorchRL components
def train_torchrl_agent(env, policy, frames_per_batch=1000, total_frames=100000, init_rand_steps=5000, 
                       batch_size=64, optim_steps=10, lr=0.001, gamma=0.99, 
                       target_update_interval=10, evaluation_interval=1000):
    # Set up collector
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        init_random_frames=init_rand_steps,
    )
    
    # Set up replay buffer
    rb = ReplayBuffer(storage=LazyTensorStorage(100_000))
    
    # Set up loss function
    loss_module = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True).to(device)
    optimizer = optim.Adam(loss_module.parameters(), lr=lr)
    updater = SoftUpdate(loss_module, eps=0.99)
    
    # Metrics for tracking
    total_frames_seen = 0
    rewards_history = []
    losses = []
    epsilon_values = []
    exploration_module = None
    
    # Find exploration module if it exists
    for module in policy.modules():
        if isinstance(module, EGreedyModule):
            exploration_module = module
            break
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    episodes_completed = 0
    episode_reward = 0
    
    for i, data in enumerate(collector):
        # Add data to replay buffer
        rb.extend(data.to(device))
        total_frames_seen += data.numel()
        
        # Calculate completed episodes and average reward
        done_mask = data["next", "done"]
        num_new_episodes = done_mask.sum().item()
        episodes_completed += num_new_episodes
        
        # Update episode_reward
        if num_new_episodes > 0:
            rewards = data["reward"].squeeze() if "reward" in data.keys() else torch.zeros(data.shape[0], device=device)
            # Compute rewards per episode
            reward_indices = torch.where(done_mask)[0]
            prev_idx = 0
            for idx in reward_indices:
                if idx >= prev_idx:  # Ensure we have rewards to sum
                    episode_reward = rewards[prev_idx:idx+1].sum().item()
                    rewards_history.append(episode_reward)
                    # Store epsilon as a float, not a tensor
                    if exploration_module:
                        eps_value = exploration_module.eps
                        # Convert to float if it's a tensor
                        if isinstance(eps_value, torch.Tensor):
                            eps_value = eps_value.item()
                        epsilon_values.append(eps_value)
                    else:
                        epsilon_values.append(0.0)
                    prev_idx = idx + 1
        
        # Training steps
        if len(rb) > batch_size:
            batch_losses = []
            for _ in range(optim_steps):
                # Sample batch
                sample = rb.sample(batch_size).to(device)
                
                # Compute loss
                loss_vals = loss_module(sample)
                loss_val = loss_vals["loss"]
                batch_losses.append(loss_val.item())  # Store as Python float
                
                # Update network
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                
                # Update exploration rate if applicable
                if exploration_module:
                    exploration_module.step(data.numel())
                
                # Update target network
                if total_frames_seen % target_update_interval == 0:
                    updater.step()
            
            losses.extend(batch_losses)
        
        # Logging
        if (i+1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(rewards_history[-10:]) if rewards_history else 0
            avg_loss = np.mean(losses[-10:]) if losses else 0
            
            # Get current epsilon value (as float)
            if exploration_module:
                eps_value = exploration_module.eps
                if isinstance(eps_value, torch.Tensor):
                    eps_value = eps_value.item()
            else:
                eps_value = 0.0
            
            print(f"Step {i+1}, Frames: {total_frames_seen}, Episodes: {episodes_completed}")
            print(f"Avg Reward: {avg_reward:.2f}, Epsilon: {eps_value:.3f}, Loss: {avg_loss:.4f}")
            print(f"Time elapsed: {elapsed_time:.1f}s\n")
        
        # Stop if we've seen enough frames
        if total_frames_seen >= total_frames:
            break
    
    total_time = time.time() - start_time
    print(f"Training completed. Total frames: {total_frames_seen}, Episodes: {episodes_completed}")
    print(f"Total time: {total_time:.2f}s")
    
    return {
        "rewards": rewards_history,
        "losses": losses,
        "epsilon_values": epsilon_values,
        "episodes": episodes_completed,
        "frames": total_frames_seen,
        "training_time": total_time
    }

# Evaluation function for TorchRL agent
def evaluate_torchrl_agent(env, policy, num_episodes=10):
    # Disable exploration during evaluation
    evaluation_policy = policy
    
    # Find and disable exploration if present
    for module in policy.modules():
        if isinstance(module, EGreedyModule):
            # Save original epsilon and set to 0 for evaluation
            original_eps = module.eps
            module.eps = 0.0
    
    rewards = []
    actions = []
    tensordict_history = []  # Store the tensor dictionaries
    total_reward = 0
    episode_count = 0
    max_steps = len(env.dataset) // env.num_keywords - 1
    
    while episode_count < num_episodes:
        # Reset environment
        tensordict = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Select action
            with torch.no_grad():
                td_next = evaluation_policy(tensordict)
                action = td_next["action"]
            
            # Move tensor to CPU before converting to numpy
            if action.is_cuda:
                action_np = action.cpu().numpy()
            else:
                action_np = action.numpy()
                
            # Record action
            actions.append(action_np)
            
            # Save current observation (for feature importance later)
            tensordict_history.append(tensordict.clone())
            
            # Take step in environment
            tensordict = env.step(td_next)
            
            # Get reward and update total
            reward = tensordict.get("reward", torch.tensor([0.0], device=device)).item()
            episode_reward += reward
            rewards.append(reward)
            
            # Check if done
            done = tensordict["done"].item()
            step += 1
        
        total_reward += episode_reward
        episode_count += 1
    
    # Restore original exploration epsilon if applicable
    for module in policy.modules():
        if isinstance(module, EGreedyModule):
            module.eps = original_eps
    
    # Convert actions to numpy array
    actions = np.array(actions)
    
    # Count action frequencies
    action_counts = np.zeros(env.action_spec.n)
    for a in actions:
        selected_action = np.argmax(a)
        action_counts[selected_action] += 1
    
    # Try to compute feature importance (simplified)
    feature_importance = None
    try:
        # Get all states from evaluation
        states = []
        for td in tensordict_history:
            # Move tensor to CPU before converting to numpy
            if td["observation"]["keyword_features"].is_cuda:
                states.append(td["observation"]["keyword_features"].cpu().numpy())
            else:
                states.append(td["observation"]["keyword_features"].numpy())
                
        states = np.array(states)
        # Flatten states across keywords
        flattened_states = states.reshape(-1, len(feature_columns))
        # Simple correlation between features and rewards as proxy for importance
        correlations = np.abs(np.corrcoef(flattened_states.T, rewards))[:len(feature_columns), -1]
        feature_importance = correlations
    except Exception as e:
        print(f"Could not compute feature importance: {e}")
    
    # Compile evaluation results
    eval_results = {
        "total_reward": total_reward,
        "avg_reward": total_reward / episode_count if episode_count > 0 else 0,
        "rewards": rewards,
        "action_counts": action_counts,
        "feature_importance": feature_importance,
        "success_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
    }
    
    return eval_results

# Main function to tie everything together
def main():
    """Main function to run the training and evaluation pipeline."""
    import time
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"ad_optimization_results_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = f"{run_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Starting digital advertising optimization pipeline...")
    print(f"Results will be saved to: {run_dir}")
    
    # Set random seeds
    set_all_seeds(42)
    
    # Generate or load dataset
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_data(1000)
    dataset_path = f"{run_dir}/synthetic_ad_data.csv"
    dataset.to_csv(dataset_path, index=False)
    print(f"Synthetic dataset saved to {dataset_path}")
    
    # Print dataset summary
    print("\nDataset summary:")
    print(f"Shape: {dataset.shape}")
    print("\nFeature stats:")
    print(dataset[feature_columns].describe().to_string())
    
    # Create environment
    env = AdOptimizationEnv(dataset, device=device)
    
    # Create policy network with TorchRL components
    flatten_module = TensorDictModule(
        FlattenInputs(),
        in_keys=[("observation", "keyword_features"), ("observation", "cash"), ("observation", "holdings")],
        out_keys=["flattened_input"]
    )
    
    # Calculate input dimensions
    feature_dim = len(feature_columns)
    num_keywords = env.num_keywords
    action_dim = env.action_spec.shape[-1]
    total_input_dim = feature_dim * num_keywords + 1 + num_keywords  # features + cash + holdings
    
    # Create MLP network
    value_mlp = MLP(
        in_features=total_input_dim, 
        out_features=action_dim, 
        num_cells=[128, 64],
        activation_class=nn.ReLU
    )
    
    # Create value network
    value_net = TensorDictModule(value_mlp, in_keys=["flattened_input"], out_keys=["action_value"])
    
    # Create policy with flattening and value network
    policy = TensorDictSequential(flatten_module, value_net, QValueModule(spec=env.action_spec))
    policy = policy.to(device)
    
    # Add exploration module
    exploration_module = EGreedyModule(
        env.action_spec, 
        annealing_num_steps=100_000, 
        eps_init=0.9,
        eps_end=0.05
    )
    exploration_module = exploration_module.to(device)
    policy_explore = TensorDictSequential(policy, exploration_module).to(device)
    
    # Train agent
    print("\nTraining RL agent...")
    training_start = time.time()
    try:
        training_metrics = train_torchrl_agent(
            env=env,
            policy=policy_explore,
            frames_per_batch=100,
            total_frames=50000,  # For faster training
            init_rand_steps=1000,
            batch_size=64,
            optim_steps=10,
            lr=0.001,
            gamma=0.99,
            target_update_interval=200
        )
        training_time = time.time() - training_start
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Generate training visualization
        print("Generating training visualization...")
        training_plot_path = visualize_training_progress(training_metrics, output_dir=plot_dir)
        print(f"Training progress plot saved to {training_plot_path}")
    except Exception as e:
        print(f"Error during training or visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    try:
        eval_episodes = 10
        eval_results = evaluate_torchrl_agent(env, policy_explore, num_episodes=eval_episodes)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print(f"Average Reward: {eval_results['avg_reward']:.2f}")
        print(f"Success Rate: {eval_results['success_rate']:.2f}")
        
        action_counts = eval_results["action_counts"]
        total_actions = action_counts.sum()
        print("\nAction Distribution:")
        for i, count in enumerate(action_counts):
            if i < env.num_keywords:
                print(f"  Keyword {i}: {count} ({100 * count / total_actions:.1f}%)")
            else:
                print(f"  No Action: {count} ({100 * count / total_actions:.1f}%)")
        
        # Save evaluation metrics
        eval_metrics_path = f"{run_dir}/evaluation_metrics.txt"
        with open(eval_metrics_path, "w") as f:
            f.write(f"Average Reward: {eval_results['avg_reward']:.4f}\n")
            f.write(f"Success Rate: {eval_results['success_rate']:.4f}\n")
            f.write(f"Total Reward: {eval_results['total_reward']:.4f}\n")
            f.write("\nAction Distribution:\n")
            for i, count in enumerate(action_counts):
                if i < env.num_keywords:
                    f.write(f"  Keyword {i}: {count} ({100 * count / total_actions:.1f}%)\n")
                else:
                    f.write(f"  No Action: {count} ({100 * count / total_actions:.1f}%)\n")
        
        # Visualize evaluation
        print("Generating evaluation visualization...")
        eval_plot_path = visualize_evaluation(eval_results, feature_columns, output_dir=plot_dir)
        print(f"Evaluation plot saved to {eval_plot_path}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Save model
    try:
        model_path = f"{run_dir}/ad_optimization_model.pt"
        torch.save({
            'model_state_dict': policy.state_dict(),
            'feature_columns': feature_columns,
            'training_metrics': training_metrics
        }, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print(f"\nPipeline completed successfully. All results saved to {run_dir}")
    return policy, training_metrics, eval_results

if __name__ == "__main__":
    main()
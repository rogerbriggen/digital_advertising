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
import copy

# TorchRL imports
from torchrl.envs import EnvBase
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.objectives import DQNLoss
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from typing import Optional, Dict, Any, List, Tuple
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.data import OneHot, Bounded, Unbounded, Binary
from torchrl.data import LazyTensorStorage, ReplayBuffer
# Using imports from torchrl.objectives.value instead

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

# Generate Realistic Synthetic Data
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
    "organic_ctr", 
    "paid_ctr", 
    "ad_spend", 
    "ad_conversions", 
    "ad_roas", 
    "conversion_rate", 
    "cost_per_click",
    "impression_share",
    "conversion_value"
]

# TorchRL Environment for Ad Optimization
class AdEnv(EnvBase):
    """A TorchRL-compliant environment for ad optimization."""
    
    def __init__(self, dataset, device="cpu"):
        super().__init__(device=device)
        
        # Store dataset
        self.dataset = dataset
        
        # Make sure numeric columns are float32
        for col in feature_columns:
            if col in self.dataset.columns:
                self.dataset[col] = self.dataset[col].astype(np.float32)
        
        # Define environment specs
        # Binary action space: 0=conservative, 1=aggressive
        self.action_spec = OneHot(n=2, device=self.device)
        
        # Simple unbounded reward
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=self.device)
        
        # Observation is just the feature columns
        self.observation_spec = Unbounded(
            shape=(len(feature_columns),), 
            dtype=torch.float32,
            device=self.device
        )
        
        # Done spec
        self.done_spec = Binary(shape=(1,), dtype=torch.bool, device=self.device)
        
        # Environment state
        self.current_index = 0
        self.max_index = len(dataset) - 1
        
        # Set a default RNG on the correct device
        self.rng = torch.Generator(device='cpu')  # Generator only supports CPU
        self.rng.manual_seed(42)
    
    def _reset(self, tensordict=None):
        """Reset the environment to the initial state."""
        self.current_index = 0
        
        # Get initial state features
        sample = self.dataset.iloc[self.current_index]
        features = sample[feature_columns].values.astype(np.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        # Create TensorDict with observation
        if tensordict is None:
            tensordict = TensorDict({
                "observation": features_tensor,
                "done": torch.tensor(False, dtype=torch.bool, device=self.device)
            }, batch_size=[], device=self.device)
        else:
            # Ensure tensordict is on the right device before updating
            if tensordict.device != self.device:
                tensordict = tensordict.to(self.device)
            
            tensordict.update({
                "observation": features_tensor,
                "done": torch.tensor(False, dtype=torch.bool, device=self.device)
            })
        
        return tensordict
    
    # Update the state
    def _step(self, tensordict):
        """Take a step in the environment based on the action."""
        # Ensure tensordict is on the right device
        if tensordict.device != self.device:
            tensordict = tensordict.to(self.device)
            
        # Get action from tensordict (one-hot encoded)
        action_onehot = tensordict["action"]
        action = torch.argmax(action_onehot).item()  # Convert to scalar (0 or 1)
        
        # Get current state features
        sample = self.dataset.iloc[self.current_index]
        
        # Calculate reward based on action and state
        reward = self._compute_reward(action, sample)
        
        # Move to next state
        self.current_index = min(self.current_index + 1, self.max_index)
        done = self.current_index >= self.max_index
        
        # Get next state features
        next_sample = self.dataset.iloc[self.current_index]
        next_features = next_sample[feature_columns].values.astype(np.float32)
        next_features_tensor = torch.tensor(next_features, dtype=torch.float32, device=self.device)
        
        # Create result TensorDict with explicit reward
        result = TensorDict({
            "observation": next_features_tensor,
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "done": torch.tensor(done, dtype=torch.bool, device=self.device),
            "terminated": torch.tensor(done, dtype=torch.bool, device=self.device),
            "next": {"observation": next_features_tensor}  # Include next observation in the next field
        }, batch_size=[], device=self.device)
        
        return result
    
    def _compute_reward(self, action, sample):
        """Compute reward based on action and current state."""
        # Extract key metrics
        cost = float(sample["ad_spend"])
        ctr = float(sample["paid_ctr"])
        revenue = float(sample["conversion_value"])
        roas = revenue / cost if cost > 0 else 0.0
        
        if action == 1:  # Aggressive strategy
            reward = 2.0 if (cost > 5000 and roas > 2.0) else (1.0 if roas > 1.0 else -1.0)
        else:  # Conservative strategy
            reward = 1.0 if ctr > 0.15 else -0.5
        
        return reward
    
    def _set_seed(self, seed: Optional[int] = None):
        """Set the seed for this environment's random number generator(s)."""
        if seed is not None:
            # Note: torch.Generator can only be on CPU
            self.rng = torch.Generator(device='cpu')
            self.rng.manual_seed(seed)
            # Also set numpy seed if you use numpy random functions
            np.random.seed(seed)
            # And Python's random if you use it
            random.seed(seed)
        return seed  # Return the seed itself, not the generator

# Create Q-network using TorchRL's MLP module
def create_q_network(input_size, output_size):
    """Create a Q-network with TorchRL's MLP module."""
    q_network = MLP(
        in_features=input_size,
        out_features=output_size,
        num_cells=[64, 64],
        activation_class=nn.ReLU
    )
    return q_network

# Visualization functions
def visualize_training_progress(metrics, output_dir="plots", window_size=20):
    """Visualize training metrics including rewards, losses, and exploration rate."""
    os.makedirs(output_dir, exist_ok=True)
    
    rewards = metrics["rewards"]
    losses = metrics["losses"]
    epsilons = metrics.get("epsilon_values", [])
    
    # Ensure tensors are converted to CPU NumPy arrays
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
    if isinstance(losses, torch.Tensor):
        losses = losses.cpu().numpy()
    if isinstance(epsilons, torch.Tensor):
        epsilons = epsilons.cpu().numpy()
    
    if len(rewards) == 0:
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
    if len(losses) > 0:
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
    if len(epsilons) > 0:
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
    feature_importance = eval_results.get("feature_importance", None)
    
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
    action_labels = ["Conservative", "Aggressive"]
    
    # Check if we have any actions
    if np.sum(action_counts) > 0:
        axs[0, 0].bar(action_labels, action_counts)
        axs[0, 0].set_title("Action Distribution")
        axs[0, 0].set_ylabel("Count")
        plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha="right")
    else:
        axs[0, 0].text(0.5, 0.5, "No actions recorded during evaluation", 
                      ha='center', va='center', fontsize=12)
        axs[0, 0].set_title("Action Distribution (Empty)")
    
    # 2. Rewards Distribution (Top Right)
    if len(rewards) > 0:
        axs[0, 1].hist(rewards, bins=min(20, len(set(rewards))), alpha=0.7, color='blue')
        mean_reward = np.mean(rewards) if len(rewards) > 0 else 0
        axs[0, 1].axvline(x=mean_reward, color='r', linestyle='--', 
                         label=f'Mean: {mean_reward:.2f}')
        axs[0, 1].set_title("Reward Distribution")
        axs[0, 1].set_xlabel("Reward")
        axs[0, 1].set_ylabel("Frequency")
        axs[0, 1].legend()
    else:
        axs[0, 1].text(0.5, 0.5, "No reward data available", 
                      ha='center', va='center', fontsize=12)
        axs[0, 1].set_title("Reward Distribution (Empty)")
    
    # 3. Feature Importance (Bottom Left)
    if feature_importance is not None and len(feature_importance) > 0 and not np.isnan(feature_importance).any():
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)
        sorted_idx = sorted_idx[::-1]  # Reverse for descending order
        axs[1, 0].barh([feature_columns[i] for i in sorted_idx], 
                      [feature_importance[i] for i in sorted_idx])
        axs[1, 0].set_title("Feature Importance")
        axs[1, 0].set_xlabel("Importance Score")
    else:
        axs[1, 0].text(0.5, 0.5, "Feature importance not available", 
                      ha='center', va='center', fontsize=12)
        axs[1, 0].set_title("Feature Importance (Not Available)")
    
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
        axs[1, 1].set_title("Performance Over Time (No Data)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/evaluation_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path

# Simplified training function with minimal TorchRL dependencies
def train_agent_simple(env, policy, total_frames=50000, batch_size=64, lr=0.001, gamma=0.99, target_update_freq=10):
    """
    Train a policy with a simplified approach that's less dependent on specific TorchRL features.
    
    Args:
        env: The environment to train on
        policy: The policy module to train
        total_frames: Total number of frames to train for
        batch_size: Batch size for training
        lr: Learning rate
        gamma: Discount factor
        target_update_freq: Frequency of target network updates
        
    Returns:
        dict: Training metrics
    """
    # Create target network
    target_policy = copy.deepcopy(policy)
    target_policy.requires_grad_(False)
    
    # Set up optimizer
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Create replay buffer
    buffer_capacity = 10000
    replay_buffer = []
    
    # Metrics for tracking
    total_frames_seen = 0
    rewards_history = []
    losses = []
    epsilon_values = []
    episodes_completed = 0
    episode_reward = 0
    
    print("Starting training...")
    start_time = time.time()
    
    # Initialize environment
    td = env.reset()
    
    # Training loop
    while total_frames_seen < total_frames:
        # For exploration, we use epsilon-greedy directly
        # Get epsilon value from exploration module if it exists
        epsilon = 0.9 * max(0.1, 1.0 - (total_frames_seen / (total_frames * 0.5)))
        
        # Select action
        with torch.no_grad():
            # Forward through policy
            td_action = policy(td)
            
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                # Random action
                random_action = torch.zeros_like(td_action["action"])
                random_idx = random.randint(0, 1)  # Binary action
                random_action[random_idx] = 1.0
                td_action["action"] = random_action
        
        # Store current state
        current_obs = td["observation"].clone()
        current_action = td_action["action"].clone()
        
        # Step environment
        td_next = env.step(td_action)
        
        # Get reward (handle missing reward key)
        if "reward" in td_next:
            reward = td_next["reward"].item()
        else:
            # Calculate reward manually if it's missing
            current_index = env.current_index - 1  # Step has already incremented this
            sample = env.dataset.iloc[current_index]
            reward = env._compute_reward(torch.argmax(current_action).item(), sample)
        
        # Check for done key
        if "done" in td_next:
            done = td_next["done"].item()
        elif "terminated" in td_next:
            done = td_next["terminated"].item()
        else:
            # Fallback: check if we've reached the end
            done = env.current_index >= env.max_index
        
        # Store transition
        replay_buffer.append({
            "observation": current_obs,
            "action": current_action,
            "next_observation": td_next["observation"].clone(),
            "reward": reward,
            "done": done
        })
        
        # Limit buffer size
        if len(replay_buffer) > buffer_capacity:
            replay_buffer.pop(0)
        
        # Update episode reward
        episode_reward += reward
        
        # If episode is done, reset environment
        if done:
            rewards_history.append(episode_reward)
            epsilon_values.append(epsilon)
            episode_reward = 0
            episodes_completed += 1
            td = env.reset()
        else:
            td = td_next
        
        # Increment frame counter
        total_frames_seen += 1
        
        # Train if we have enough data
        if len(replay_buffer) >= batch_size:
            # Sample batch
            batch_indices = random.sample(range(len(replay_buffer)), batch_size)
            batch = [replay_buffer[idx] for idx in batch_indices]
            
            # Convert batch to tensors
            obs_batch = torch.stack([item["observation"] for item in batch])
            act_batch = torch.stack([item["action"] for item in batch])
            next_obs_batch = torch.stack([item["next_observation"] for item in batch])
            rewards_batch = torch.tensor([item["reward"] for item in batch], device=device)
            done_batch = torch.tensor([item["done"] for item in batch], dtype=torch.bool, device=device)
            
            # Compute Q-values for current states and actions
            q_values = policy(TensorDict({"observation": obs_batch}, batch_size=[batch_size]))["action_value"]
            action_indices = torch.argmax(act_batch, dim=1, keepdim=True)
            q_values_selected = q_values.gather(1, action_indices).squeeze()
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = target_policy(TensorDict({"observation": next_obs_batch}, batch_size=[batch_size]))["action_value"]
                next_q_values_max = next_q_values.max(1)[0]
                target_q_values = rewards_batch + gamma * next_q_values_max * (~done_batch)
            
            # Compute loss
            loss = nn.MSELoss()(q_values_selected, target_q_values)
            
            # Update policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Update target network periodically
        if total_frames_seen % target_update_freq == 0:
            target_policy.load_state_dict(policy.state_dict())
        
        # Logging
        if total_frames_seen % 1000 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(rewards_history[-10:]) if rewards_history else 0
            avg_loss = np.mean(losses[-10:]) if losses else 0
            
            print(f"Frame {total_frames_seen}/{total_frames}, Episodes: {episodes_completed}")
            print(f"Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Loss: {avg_loss:.4f}")
            print(f"Time elapsed: {elapsed_time:.1f}s\n")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Total frames: {total_frames_seen}, Episodes: {episodes_completed}")
    
    return {
        "rewards": rewards_history,
        "losses": losses,
        "epsilon_values": epsilon_values,
        "episodes": episodes_completed,
        "frames": total_frames_seen,
        "training_time": total_time
    }

# Simplified evaluation function
def evaluate_agent_simple(env, policy, num_episodes=10):
    """
    Evaluate a trained policy without exploration.
    
    Args:
        env: The environment to evaluate on
        policy: The policy to evaluate
        num_episodes: Number of episodes to evaluate
        
    Returns:
        dict: Evaluation metrics
    """
    rewards = []
    actions = []
    observations = []
    episode_rewards = []
    
    for _ in range(num_episodes):
        episode_reward = 0
        td = env.reset()
        done = False
        
        while not done:
            # Select action deterministically without exploration
            with torch.no_grad():
                # Forward through policy
                td_action = policy(td)
                action = td_action["action"]
                action_idx = torch.argmax(action).item()
                
                # Record for analysis
                actions.append(action_idx)
                observations.append(td["observation"].cpu().numpy())
                
                # Step environment
                td_next = env.step(td_action)
                
                # Check for reward key and handle it
                if "reward" in td_next:
                    reward = td_next["reward"].item()
                else:
                    # If reward key is missing, calculate it directly
                    current_index = env.current_index - 1  # Step has already incremented this
                    sample = env.dataset.iloc[current_index]
                    reward = env._compute_reward(action_idx, sample)
                
                rewards.append(reward)
                episode_reward += reward
                
                # Check for done key
                if "done" in td_next:
                    done = td_next["done"].item()
                elif "terminated" in td_next:
                    done = td_next["terminated"].item()
                else:
                    # Fallback: check if we've reached the end
                    done = env.current_index >= env.max_index
                
                # Update state
                td = td_next
        
        episode_rewards.append(episode_reward)
    
    # Calculate action distribution
    action_counts = np.bincount(np.array(actions), minlength=2)
    
    # Calculate feature importance
    feature_importance = None
    if observations and len(observations) == len(actions):
        obs_array = np.array(observations)
        actions_array = np.array(actions)
        feature_importance = np.zeros(len(feature_columns))
        
        for i in range(len(feature_columns)):
            feature_values = obs_array[:, i]
            # Calculate absolute correlation between feature and action choice
            if np.var(feature_values) > 0 and np.var(actions_array) > 0:
                try:
                    correlation = np.corrcoef(feature_values, actions_array)[0, 1]
                    feature_importance[i] = abs(correlation) if not np.isnan(correlation) else 0
                except:
                    feature_importance[i] = 0
    
    # Calculate success rate (positive rewards)
    success_rate = np.mean([r > 0 for r in rewards]) if rewards else 0
    
    return {
        "avg_reward": np.mean(episode_rewards),
        "total_reward": sum(episode_rewards),
        "rewards": rewards,
        "action_counts": action_counts,
        "feature_importance": feature_importance,
        "success_rate": success_rate
    }

# Main function
def main():
    """Main function to run the training and evaluation pipeline."""
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
    
    # Generate dataset
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
    print("\nCreating environment...")
    env = AdEnv(dataset, device=device)
    
    # Create a simpler policy network
    print("Creating policy network...")
    input_size = len(feature_columns)
    output_size = 2  # Binary action: Conservative or Aggressive
    
    # Create Q-network using a simple MLP
    q_network = create_q_network(input_size, output_size)
    
    # Wrap in TensorDictModule for TorchRL compatibility
    policy = TensorDictModule(
        q_network,
        in_keys=["observation"],
        out_keys=["action_value"]
    )
    
    # Add QValueModule to convert to actions
    policy = TensorDictSequential(policy, QValueModule(spec=env.action_spec))
    
    # Move to correct device
    policy = policy.to(device)
    
    # Train agent
    print("\nTraining agent...")
    training_metrics = train_agent_simple(
        env=env,
        policy=policy,
        total_frames=20000,  # Reduced for faster training
        batch_size=64,
        lr=0.001,
        gamma=0.99,
        target_update_freq=10
    )
    
    # Generate training visualization
    print("Generating training visualization...")
    training_plot_path = visualize_training_progress(training_metrics, output_dir=plot_dir)
    print(f"Training plot saved to {training_plot_path}")
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent_simple(env, policy, num_episodes=10)
    
    # Display evaluation results
    print("\nEvaluation Results:")
    print(f"Average Reward: {eval_results['avg_reward']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']:.2f}")
    
    # Print action distribution
    action_counts = eval_results["action_counts"]
    total_actions = action_counts.sum()
    
    print("\nAction Distribution:")
    if total_actions > 0:
        print(f"  Conservative: {action_counts[0]} ({100 * action_counts[0] / total_actions:.1f}%)")
        print(f"  Aggressive: {action_counts[1]} ({100 * action_counts[1] / total_actions:.1f}%)")
    else:
        print("  No actions recorded during evaluation")
    
    # Save evaluation metrics
    eval_metrics_path = f"{run_dir}/evaluation_metrics.txt"
    with open(eval_metrics_path, "w") as f:
        f.write(f"Average Reward: {eval_results['avg_reward']:.4f}\n")
        f.write(f"Success Rate: {eval_results['success_rate']:.4f}\n")
        f.write(f"Total Reward: {eval_results['total_reward']:.4f}\n")
        f.write("\nAction Distribution:\n")
        
        if total_actions > 0:
            f.write(f"  Conservative: {action_counts[0]} ({100 * action_counts[0] / total_actions:.1f}%)\n")
            f.write(f"  Aggressive: {action_counts[1]} ({100 * action_counts[1] / total_actions:.1f}%)\n")
        else:
            f.write("  No actions recorded during evaluation\n")
    
    # Visualize evaluation
    print("Generating evaluation visualization...")
    eval_plot_path = visualize_evaluation(eval_results, feature_columns, output_dir=plot_dir)
    print(f"Evaluation plot saved to {eval_plot_path}")
    
    # Save model
    model_path = f"{run_dir}/ad_optimization_model.pt"
    torch.save({
        'model_state_dict': policy.state_dict(),
        'feature_columns': feature_columns,
        'training_metrics': training_metrics
    }, model_path)
    print(f"Model saved to {model_path}")
    
    print(f"\nPipeline completed successfully. All results saved to {run_dir}")
    
    return policy, training_metrics, eval_results

if __name__ == "__main__":
    main()
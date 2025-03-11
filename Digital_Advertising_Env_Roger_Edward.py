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

# Simple Ad Environment
class AdEnv:
    def __init__(self, dataset):
        # Ensure all numeric columns are float32
        self.dataset = dataset.copy()
        for col in feature_columns:
            self.dataset[col] = self.dataset[col].astype(np.float32)
            
        self.feature_columns = feature_columns
        self.num_features = len(self.feature_columns)
        self.current_index = 0
        self.max_index = len(dataset) - 1
        
    def reset(self):
        """Reset environment and return initial state."""
        self.current_index = 0
        sample = self.dataset.iloc[self.current_index]
        
        # Convert feature values to a float32 numpy array
        features = sample[self.feature_columns].values.astype(np.float32)
        state = torch.tensor(features, dtype=torch.float32, device=device)
        
        return state
    
    def step(self, action):
        """Execute one step in the environment."""
        # Get current state
        sample = self.dataset.iloc[self.current_index]
        
        # Calculate reward
        reward = self._compute_reward(action, sample)
        
        # Move to next state
        self.current_index = min(self.current_index + 1, self.max_index)
        next_sample = self.dataset.iloc[self.current_index]
        
        # Convert feature values to float32 numpy array
        next_features = next_sample[self.feature_columns].values.astype(np.float32)
        next_state = torch.tensor(next_features, dtype=torch.float32, device=device)
        
        # Check if episode is done
        done = self.current_index >= self.max_index
        
        return next_state, reward, done
    
    def _compute_reward(self, action, sample):
        """Compute reward based on action and current state."""
        cost = float(sample["ad_spend"])
        ctr = float(sample["paid_ctr"])
        revenue = float(sample["conversion_value"])
        roas = revenue / cost if cost > 0 else 0.0
        
        if action == 1:  # Aggressive strategy
            reward = 2.0 if (cost > 5000 and roas > 2.0) else (1.0 if roas > 1.0 else -1.0)
        else:  # Conservative strategy
            reward = 1.0 if ctr > 0.15 else -0.5
        
        return reward

# Simple Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(states), 
                torch.tensor(actions, dtype=torch.long, device=device), 
                torch.tensor(rewards, dtype=torch.float, device=device),
                torch.stack(next_states), 
                torch.tensor(dones, dtype=torch.bool, device=device))
    
    def __len__(self):
        return len(self.buffer)

# Agent with epsilon-greedy exploration
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        
    def select_action(self, state, eval_mode=False):
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.action_size)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Get current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Training function
def train_dqn(env, agent, num_episodes=500, batch_size=64, target_update=10, buffer_size=10000):
    replay_buffer = ReplayBuffer(buffer_size)
    rewards_history = []
    losses = []
    epsilon_values = []
    
    print("Starting training...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Train agent
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.train(batch)
                episode_loss += loss
                step_count += 1
            
            episode_reward += reward
            
            # Update target network
            if step_count % target_update == 0 and step_count > 0:
                agent.update_target_network()
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record metrics
        rewards_history.append(episode_reward)
        epsilon_values.append(agent.epsilon)
        if step_count > 0:
            losses.append(episode_loss / step_count)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    print("Training completed!")
    
    return {
        "rewards": rewards_history,
        "losses": losses,
        "epsilon_values": epsilon_values
    }

# Evaluation function
def evaluate_agent(env, agent, num_episodes=10):
    total_reward = 0
    episode_lengths = []
    action_counts = {0: 0, 1: 0}  # 0=conservative, 1=aggressive
    decisions = []
    rewards = []
    states = []
    conservative_rewards = []
    aggressive_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get action using greedy policy
            action = agent.select_action(state, eval_mode=True)
            
            # Record state
            states.append(state.cpu().numpy())
            action_counts[action] += 1
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Record results
            decisions.append((action, reward))
            rewards.append(reward)
            
            if action == 0:
                conservative_rewards.append(reward)
            else:
                aggressive_rewards.append(reward)
            
            episode_reward += reward
            steps += 1
            
            # Update state
            state = next_state
        
        total_reward += episode_reward
        episode_lengths.append(steps)
    
    # Calculate metrics
    avg_reward = total_reward / num_episodes if num_episodes > 0 else 0
    avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
    
    total_actions = sum(action_counts.values())
    action_distribution = {k: v / total_actions for k, v in action_counts.items()} if total_actions > 0 else {0: 0, 1: 0}
    
    avg_conservative_reward = np.mean(conservative_rewards) if conservative_rewards else 0
    avg_aggressive_reward = np.mean(aggressive_rewards) if aggressive_rewards else 0
    
    success_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
    
    return {
        "avg_reward": avg_reward,
        "avg_episode_length": avg_episode_length,
        "action_distribution": action_distribution,
        "decisions": decisions,
        "rewards": rewards,
        "states": np.array(states) if states else np.array([]),
        "avg_conservative_reward": avg_conservative_reward,
        "avg_aggressive_reward": avg_aggressive_reward,
        "success_rate": success_rate
    }

# Visualization functions
def visualize_training_progress(metrics, output_dir="plots", window_size=20):
    """Visualize training metrics including rewards, losses, and exploration rate."""
    os.makedirs(output_dir, exist_ok=True)
    
    rewards = metrics["rewards"]
    losses = metrics["losses"]
    epsilons = metrics["epsilon_values"]
    
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

def visualize_evaluation(metrics, feature_columns, output_dir="plots"):
    """Create visualizations for evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set(style="whitegrid")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Ad Optimization RL Agent Evaluation", fontsize=16)
    
    # 1. Action Distribution
    ax1 = fig.add_subplot(2, 3, 1)
    actions = ["Conservative", "Aggressive"]
    action_counts = [metrics["action_distribution"].get(0, 0), metrics["action_distribution"].get(1, 0)]
    ax1.bar(actions, action_counts, color=["skyblue", "coral"])
    ax1.set_title("Action Distribution")
    ax1.set_ylabel("Frequency")
    
    # 2. Average Reward by Action Type
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(["Conservative", "Aggressive"], 
            [metrics["avg_conservative_reward"], metrics["avg_aggressive_reward"]], 
            color=["skyblue", "coral"])
    ax2.set_title("Average Reward by Action Type")
    ax2.set_ylabel("Average Reward")
    
    # 3. Feature Correlations with Decisions
    ax3 = fig.add_subplot(2, 3, 3)
    states = np.array(metrics["states"])
    decisions = np.array([a for a, _ in metrics["decisions"]])
    
    correlations = []
    feature_names = []
    
    if states.size > 0 and decisions.size > 0 and states.shape[1] == len(feature_columns):
        for i, feature in enumerate(feature_columns):
            try:
                corr = np.corrcoef(states[:, i], decisions)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    feature_names.append(feature)
            except:
                pass
    
    if correlations:
        sorted_indices = np.argsort(np.abs(correlations))[::-1][:5]  # Top 5 features
        top_features = [feature_names[i] for i in sorted_indices]
        top_correlations = [correlations[i] for i in sorted_indices]
        
        ax3.barh(top_features, top_correlations, color='teal')
        ax3.set_title("Top Feature Correlations with Actions")
        ax3.set_xlabel("Correlation Coefficient")
    else:
        ax3.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                ha='center', va='center')
    
    # 4. Reward Distribution
    ax4 = fig.add_subplot(2, 3, 4)
    if metrics["rewards"]:
        sns.histplot(metrics["rewards"], kde=True, ax=ax4)
        ax4.set_title("Reward Distribution")
        ax4.set_xlabel("Reward")
        ax4.set_ylabel("Frequency")
    else:
        ax4.text(0.5, 0.5, "No reward data available", ha='center', va='center')
    
    # 5. Decision Quality Matrix
    ax5 = fig.add_subplot(2, 3, 5)
    decision_quality = np.zeros((2, 2))
    
    for action, reward in metrics["decisions"]:
        quality = 1 if reward > 0 else 0
        if action < 2:  # Ensure action is either 0 or 1
            decision_quality[action, quality] += 1
    
    row_sums = decision_quality.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    decision_quality_norm = decision_quality / row_sums
    
    sns.heatmap(decision_quality_norm, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=["Poor", "Good"], 
                yticklabels=["Conservative", "Aggressive"],
                ax=ax5)
    ax5.set_title("Decision Quality Matrix")
    ax5.set_ylabel("Action")
    ax5.set_xlabel("Decision Quality")
    
    # 6. Success Rate Over Time
    ax6 = fig.add_subplot(2, 3, 6)
    if metrics["decisions"]:
        # Calculate moving success rate
        window = min(20, len(metrics["decisions"]))
        success_rates = []
        for i in range(len(metrics["decisions"]) - window + 1):
            window_decisions = metrics["decisions"][i:i+window]
            success_rate = sum(1 for _, r in window_decisions if r > 0) / window
            success_rates.append(success_rate)
        
        ax6.plot(range(window-1, len(metrics["decisions"])), success_rates, color='green')
        ax6.set_title(f"Success Rate (Moving Window: {window})")
        ax6.set_xlabel("Decision")
        ax6.set_ylabel("Success Rate")
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "Insufficient data for success rate analysis", 
                ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/agent_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path

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
    env = AdEnv(dataset)
    
    # Create agent
    agent = DQNAgent(state_size=len(feature_columns), action_size=2, learning_rate=0.001)
    
    # Train agent
    print("\nTraining RL agent...")
    training_metrics = train_dqn(env, agent, num_episodes=200, batch_size=64)
    
    # Check if we have training data to plot
    if training_metrics["rewards"]:
        # Save training metrics plot
        print("Generating training visualization...")
        training_plot_path = visualize_training_progress(training_metrics, output_dir=plot_dir)
        print(f"Training progress plot saved to {training_plot_path}")
    else:
        print("Warning: No training rewards collected, skipping training visualization")
    
    # Evaluate agent
    print("Evaluating trained agent...")
    eval_episodes = 10
    eval_metrics = evaluate_agent(env, agent, num_episodes=eval_episodes)
    
    # Save evaluation metrics
    eval_metrics_path = f"{run_dir}/evaluation_metrics.txt"
    with open(eval_metrics_path, "w") as f:
        f.write(f"Average Reward: {eval_metrics['avg_reward']:.4f}\n")
        f.write(f"Success Rate: {eval_metrics['success_rate']:.4f}\n")
        f.write(f"Action Distribution: Conservative: {eval_metrics['action_distribution'].get(0, 0):.2f}, " + 
                f"Aggressive: {eval_metrics['action_distribution'].get(1, 0):.2f}\n")
        f.write(f"Average Conservative Reward: {eval_metrics['avg_conservative_reward']:.4f}\n")
        f.write(f"Average Aggressive Reward: {eval_metrics['avg_aggressive_reward']:.4f}\n")
    
    # Visualize evaluation
    print("Generating evaluation visualization...")
    eval_plot_path = visualize_evaluation(eval_metrics, feature_columns, output_dir=plot_dir)
    print(f"Evaluation plot saved to {eval_plot_path}")
    
    # Save model
    model_path = f"{run_dir}/ad_optimization_model.pt"
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'feature_columns': feature_columns,
        'training_metrics': training_metrics
    }, model_path)
    print(f"Model saved to {model_path}")
    
    print(f"Pipeline completed successfully. All results saved to {run_dir}")
    return agent, training_metrics, eval_metrics

if __name__ == "__main__":
    main()
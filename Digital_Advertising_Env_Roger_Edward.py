#!/usr/bin/env python
# coding: utf-8

# IMPORTS SECTION
# ------------------------------------------------------
import torch                # PyTorch deep learning library
import torch.nn as nn      # Neural network modules
import torch.optim as optim # Optimization algorithms
import numpy as np         # Numerical computing
import pandas as pd        # Data manipulation and analysis
import random              # Random number generation
import matplotlib.pyplot as plt # Plotting
import seaborn as sns      # Statistical data visualization
from datetime import datetime # Date and time utilities
import os                  # Operating system interfaces
from collections import deque # Double-ended queue for replay buffer

# SETUP SECTION
# ------------------------------------------------------
# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)             # Python's built-in random
    np.random.seed(seed)          # NumPy's random
    torch.manual_seed(seed)       # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # PyTorch GPU single device
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU multi-device
    return seed

# DATA GENERATION SECTION
# ------------------------------------------------------
def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic advertising data with realistic correlations.
    This function creates a dataset that mimics real-world digital advertising metrics
    with appropriate distributions and correlations between variables.
    
    Args:
        num_samples (int): Number of data points to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing synthetic advertising data
    """
    # Generate a base difficulty distribution using beta distribution
    # Beta distribution is good for values between 0 and 1 with controlled shape
    base_difficulty = np.random.beta(2.5, 3.5, num_samples)
    
    # Initialize data dictionary with random values for all features
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
    
    # Add realistic correlations between features
    # Difficulty score is influenced by competitiveness and base difficulty
    data["difficulty_score"] = 0.7 * data["competitiveness"] + 0.3 * base_difficulty
    
    # Organic rank (position in search results) depends on difficulty score
    # Higher difficulty = worse rank (higher number)
    data["organic_rank"] = 1 + np.floor(9 * data["difficulty_score"] + np.random.normal(0, 1, num_samples).clip(-2, 2))
    data["organic_rank"] = data["organic_rank"].clip(1, 10).astype(int)  # Ensure ranks are integers between 1-10
    
    # CTR (Click-Through Rate) follows a beta distribution and negatively correlates with rank
    # Better ranks (lower numbers) get higher CTRs
    base_ctr = np.random.beta(1.5, 10, num_samples)  # Generate base CTR distribution
    rank_effect = (11 - data["organic_rank"]) / 10   # Transform rank to 0.1-1.0 scale (higher for better ranks)
    data["organic_ctr"] = (base_ctr * rank_effect * 0.3).clip(0.01, 0.3)  # Apply rank effect and scale to realistic range
    
    # Organic clicks based on CTR and a base impression count
    # Using lognormal distribution for impressions (common in web traffic)
    base_impressions = np.random.lognormal(8, 1, num_samples).astype(int)
    data["organic_clicks"] = (base_impressions * data["organic_ctr"]).astype(int)
    
    # Paid CTR correlates with organic CTR but with more variance
    data["paid_ctr"] = (data["organic_ctr"] * np.random.normal(1, 0.3, num_samples)).clip(0.01, 0.25)
    
    # Paid clicks calculation
    paid_impressions = np.random.lognormal(7, 1.2, num_samples).astype(int)
    data["paid_clicks"] = (paid_impressions * data["paid_ctr"]).astype(int)
    
    # Cost per click is higher for more competitive keywords
    data["cost_per_click"] = (0.5 + 9.5 * data["competitiveness"] * np.random.normal(1, 0.2, num_samples)).clip(0.1, 10)
    
    # Ad spend calculation based on CPC and clicks
    data["ad_spend"] = data["paid_clicks"] * data["cost_per_click"]
    
    # Conversion rate with realistic e-commerce distribution (typically low, around 1-3%)
    data["conversion_rate"] = np.random.beta(1.2, 15, num_samples).clip(0.01, 0.3)
    
    # Ad conversions calculation
    data["ad_conversions"] = (data["paid_clicks"] * data["conversion_rate"]).astype(int)
    
    # Conversion value with variance (using lognormal for realistic distribution of order values)
    base_value = np.random.lognormal(4, 1, num_samples)
    data["conversion_value"] = data["ad_conversions"] * base_value
    
    # Cost per acquisition calculation
    # Using np.errstate to ignore division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        data["cost_per_acquisition"] = np.where(
            data["ad_conversions"] > 0, 
            data["ad_spend"] / data["ad_conversions"], 
            500  # Default high CPA for no conversions
        ).clip(5, 500)
    
    # ROAS (Return on Ad Spend) calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        data["ad_roas"] = np.where(
            data["ad_spend"] > 0,
            data["conversion_value"] / data["ad_spend"],
            0
        ).clip(0.5, 5)
    
    # Impression share - competitive keywords have lower share
    data["impression_share"] = (1 - 0.6 * data["competitiveness"] * np.random.normal(1, 0.2, num_samples)).clip(0.1, 1.0)
    
    return pd.DataFrame(data)

# Define feature columns to use in the environment
# These are the metrics that will be used as state representation for the RL agent
feature_columns = [
    "competitiveness",    # How competitive the keyword is
    "difficulty_score",   # Difficulty to rank for this keyword
    "organic_rank",       # Position in organic search results
    "organic_clicks",     # Number of clicks from organic search
    "organic_ctr",        # Click-through rate for organic results
    "paid_clicks",        # Number of clicks from paid ads
    "paid_ctr",           # Click-through rate for paid ads
    "ad_spend",           # Total money spent on ads
    "ad_conversions",     # Number of conversions from ads
    "ad_roas",            # Return on ad spend
    "conversion_rate",    # Percentage of clicks that convert
    "cost_per_click"      # Average cost per click
]

# ENVIRONMENT SECTION
# ------------------------------------------------------
class AdEnv:
    """
    Digital Advertising Environment for Reinforcement Learning.
    
    This class implements a simplified environment that simulates
    digital advertising decisions and outcomes. It follows the
    standard RL environment interface with reset() and step() methods.
    """
    def __init__(self, dataset):
        """
        Initialize the environment with a dataset of advertising metrics.
        
        Args:
            dataset (pandas.DataFrame): Dataset containing advertising metrics
        """
        # Ensure all numeric columns are float32 for compatibility with PyTorch
        self.dataset = dataset.copy()
        for col in feature_columns:
            self.dataset[col] = self.dataset[col].astype(np.float32)
            
        self.feature_columns = feature_columns
        self.num_features = len(self.feature_columns)
        self.current_index = 0
        self.max_index = len(dataset) - 1
        
    def reset(self):
        """
        Reset environment and return initial state.
        
        Returns:
            torch.Tensor: Initial state tensor
        """
        self.current_index = 0
        sample = self.dataset.iloc[self.current_index]
        
        # Convert feature values to a float32 numpy array
        features = sample[self.feature_columns].values.astype(np.float32)
        state = torch.tensor(features, dtype=torch.float32, device=device)
        
        return state
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action to take (0=conservative, 1=aggressive)
            
        Returns:
            tuple: (next_state, reward, done)
                - next_state (torch.Tensor): Next state after taking action
                - reward (float): Reward for taking action
                - done (bool): Whether episode is done
        """
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
        """
        Compute reward based on action and current state.
        
        Args:
            action (int): Action taken (0=conservative, 1=aggressive)
            sample (pandas.Series): Current state
            
        Returns:
            float: Reward value
        """
        cost = float(sample["ad_spend"])
        ctr = float(sample["paid_ctr"])
        revenue = float(sample["conversion_value"])
        roas = revenue / cost if cost > 0 else 0.0
        
        if action == 1:  # Aggressive strategy
            # For aggressive strategy:
            # - High reward if high spend WITH high ROAS (good performance)
            # - Moderate reward if profitable (ROAS > 1)
            # - Penalty if not profitable
            reward = 2.0 if (cost > 5000 and roas > 2.0) else (1.0 if roas > 1.0 else -1.0)
        else:  # Conservative strategy
            # For conservative strategy:
            # - Reward for high CTR (good engagement)
            # - Small penalty otherwise
            reward = 1.0 if ctr > 0.15 else -0.5
        
        return reward

# MODEL SECTION
# ------------------------------------------------------
class QNetwork(nn.Module):
    """
    Q-Network for Deep Q-Learning.
    
    A simple neural network that predicts Q-values for each action
    given a state input.
    """
    def __init__(self, input_size, output_size):
        """
        Initialize Q-Network.
        
        Args:
            input_size (int): Size of the input state vector
            output_size (int): Number of possible actions
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)   # First fully connected layer
        self.fc2 = nn.Linear(64, 64)           # Second fully connected layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer (one value per action)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x = self.fc3(x)              # Linear output for Q-values
        return x

# EXPERIENCE REPLAY BUFFER
# ------------------------------------------------------
class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    
    Stores (state, action, reward, next_state, done) tuples and
    allows random sampling for experience replay in DQN training.
    """
    def __init__(self, capacity):
        """
        Initialize buffer with fixed capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)  # Fixed-size buffer
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state (torch.Tensor): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (torch.Tensor): Next state
            done (bool): Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        # Sample random transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to appropriate tensor types and move to device
        return (torch.stack(states), 
                torch.tensor(actions, dtype=torch.long, device=device), 
                torch.tensor(rewards, dtype=torch.float, device=device),
                torch.stack(next_states), 
                torch.tensor(dones, dtype=torch.bool, device=device))
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)

# AGENT SECTION
# ------------------------------------------------------
class DQNAgent:
    """
    Deep Q-Network Agent with epsilon-greedy exploration.
    
    Implements the DQN algorithm with experience replay and
    target network for stable learning.
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initialize DQN agent.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Policy network (main network for taking actions)
        self.policy_net = QNetwork(state_size, action_size).to(device)
        
        # Target network (for stable learning targets)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Initialize with same weights
        
        # Optimizer for updating policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Exploration parameters
        self.epsilon = 1.0        # Initial exploration rate (100%)
        self.epsilon_min = 0.05   # Minimum exploration rate (5%)
        self.epsilon_decay = 0.995  # Decay rate per episode
        
        # Discount factor for future rewards
        self.gamma = 0.99
        
    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state
            eval_mode (bool): If True, use greedy policy (no exploration)
            
        Returns:
            int: Selected action
        """
        # Use greedy policy if in evaluation mode or random number > epsilon
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                # Choose action with highest Q-value
                return self.policy_net(state).argmax().item()
        else:
            # Choose random action for exploration
            return random.randrange(self.action_size)
    
    def update_epsilon(self):
        """Update exploration rate by decay factor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, batch):
        """
        Train the agent using a batch of experiences.
        
        Args:
            batch (tuple): Batch of (states, actions, rewards, next_states, dones)
            
        Returns:
            float: Loss value
        """
        states, actions, rewards, next_states, dones = batch
        
        # Get current Q values for actions taken
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values using target network (Double DQN approach)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            # if done, use only the immediate reward
            target_q = rewards + self.gamma * next_q * (~dones)
        
        # Compute MSE loss between current and target Q values
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize the policy network
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()            # Compute gradients
        self.optimizer.step()      # Update weights
        
        return loss.item()

# TRAINING FUNCTION
# ------------------------------------------------------
def train_dqn(env, agent, num_episodes=500, batch_size=64, target_update=10, buffer_size=10000):
    """
    Train a DQN agent in the advertising environment.
    
    Args:
        env (AdEnv): Advertising environment
        agent (DQNAgent): DQN agent to train
        num_episodes (int): Number of training episodes
        batch_size (int): Batch size for training
        target_update (int): Steps between target network updates
        buffer_size (int): Size of replay buffer
        
    Returns:
        dict: Training metrics including rewards, losses, and epsilon values
    """
    replay_buffer = ReplayBuffer(buffer_size)
    rewards_history = []   # Track episode rewards
    losses = []            # Track training losses
    epsilon_values = []    # Track exploration rate
    
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
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Train agent if enough samples in buffer
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.train(batch)
                episode_loss += loss
                step_count += 1
            
            episode_reward += reward
            
            # Update target network periodically
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

# EVALUATION FUNCTION
# ------------------------------------------------------
def evaluate_agent(env, agent, num_episodes=10):
    """
    Evaluate a trained agent in the advertising environment.
    
    Args:
        env (AdEnv): Advertising environment
        agent (DQNAgent): Trained DQN agent to evaluate
        num_episodes (int): Number of evaluation episodes
        
    Returns:
        dict: Evaluation metrics
    """
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
            # Get action using greedy policy (no exploration)
            action = agent.select_action(state, eval_mode=True)
            
            # Record state
            states.append(state.cpu().numpy())
            action_counts[action] += 1
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Record results
            decisions.append((action, reward))
            rewards.append(reward)
            
            # Separate rewards by action type for analysis
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

# VISUALIZATION FUNCTIONS
# ------------------------------------------------------
def visualize_training_progress(metrics, output_dir="plots", window_size=20):
    """
    Visualize training metrics including rewards, losses, and exploration rate.
    
    Args:
        metrics (dict): Training metrics
        output_dir (str): Directory to save plots
        window_size (int): Window size for moving average
        
    Returns:
        str: Path to saved plot
    """
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
        # Add smoothed rewards line using moving average
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
    """
    Create visualizations for evaluation metrics.
    
    Args:
        metrics (dict): Evaluation metrics
        feature_columns (list): List of feature column names
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set(style="whitegrid")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Ad Optimization RL Agent Evaluation", fontsize=16)
    
    # 1. Action Distribution - Shows proportion of conservative vs aggressive actions
    ax1 = fig.add_subplot(2, 3, 1)
    actions = ["Conservative", "Aggressive"]
    action_counts = [metrics["action_distribution"].get(0, 0), metrics["action_distribution"].get(1, 0)]
    ax1.bar(actions, action_counts, color=["skyblue", "coral"])
    ax1.set_title("Action Distribution")
    ax1.set_ylabel("Frequency")
    
    # 2. Average Reward by Action Type - Compares performance of each strategy
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(["Conservative", "Aggressive"], 
            [metrics["avg_conservative_reward"], metrics["avg_aggressive_reward"]], 
            color=["skyblue", "coral"])
    ax2.set_title("Average Reward by Action Type")
    ax2.set_ylabel("Average Reward")
    
    # 3. Feature Correlations with Decisions - Shows which features influenced action choices
    ax3 = fig.add_subplot(2, 3, 3)
    states = np.array(metrics["states"])
    decisions = np.array([a for a, _ in metrics["decisions"]])
    
    correlations = []
    feature_names = []
    
    # Calculate correlations between features and decisions
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
        # Show top 5 most influential features
        sorted_indices = np.argsort(np.abs(correlations))[::-1][:5]
        top_features = [feature_names[i] for i in sorted_indices]
        top_correlations = [correlations[i] for i in sorted_indices]
        
        ax3.barh(top_features, top_correlations, color='teal')
        ax3.set_title("Top Feature Correlations with Actions")
        ax3.set_xlabel("Correlation Coefficient")
    else:
        ax3.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                ha='center', va='center')
    
    # 4. Reward Distribution - Shows overall distribution of rewards
    ax4 = fig.add_subplot(2, 3, 4)
    if metrics["rewards"]:
        sns.histplot(metrics["rewards"], kde=True, ax=ax4)
        ax4.set_title("Reward Distribution")
        ax4.set_xlabel("Reward")
        ax4.set_ylabel("Frequency")
    else:
        ax4.text(0.5, 0.5, "No reward data available", ha='center', va='center')
    
    # 5. Decision Quality Matrix - Heatmap showing how often each action leads to good/poor outcomes
    ax5 = fig.add_subplot(2, 3, 5)
    decision_quality = np.zeros((2, 2))
    
    # Count instances of (action, outcome quality) pairs
    for action, reward in metrics["decisions"]:
        quality = 1 if reward > 0 else 0  # 1 = good outcome (positive reward), 0 = poor outcome
        if action < 2:  # Ensure action is either 0 or 1
            decision_quality[action, quality] += 1
    
    # Normalize by row to show percentages
    row_sums = decision_quality.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    decision_quality_norm = decision_quality / row_sums
    
    # Create heatmap
    sns.heatmap(decision_quality_norm, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=["Poor", "Good"], 
                yticklabels=["Conservative", "Aggressive"],
                ax=ax5)
    ax5.set_title("Decision Quality Matrix")
    ax5.set_ylabel("Action")
    ax5.set_xlabel("Decision Quality")
    
    # 6. Success Rate Over Time - Shows if agent improves over time
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

# MAIN EXECUTION FUNCTION
# ------------------------------------------------------
def main():
    """
    Main function to run the training and evaluation pipeline.
    
    This function:
    1. Creates output directories
    2. Generates synthetic data
    3. Initializes environment and agent
    4. Trains the agent
    5. Evaluates the agent
    6. Creates visualizations
    7. Saves results
    """
    # Create output directory with timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"ad_optimization_results_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = f"{run_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Starting digital advertising optimization pipeline...")
    print(f"Results will be saved to: {run_dir}")
    
    # Set random seeds for reproducibility
    set_all_seeds(42)
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_data(1000)  # Generate 1000 samples
    dataset_path = f"{run_dir}/synthetic_ad_data.csv"
    dataset.to_csv(dataset_path, index=False)
    print(f"Synthetic dataset saved to {dataset_path}")
    
    # Print dataset summary statistics
    print("\nDataset summary:")
    print(f"Shape: {dataset.shape}")
    print("\nFeature stats:")
    print(dataset[feature_columns].describe().to_string())
    
    # Create environment using the generated dataset
    env = AdEnv(dataset)
    
    # Create RL agent
    agent = DQNAgent(state_size=len(feature_columns), action_size=2, learning_rate=0.001)
    
    # Train the agent
    print("\nTraining RL agent...")
    training_metrics = train_dqn(env, agent, num_episodes=200, batch_size=64)
    
    # Check if we have training data to plot
    if training_metrics["rewards"]:
        # Save training metrics visualization
        print("Generating training visualization...")
        training_plot_path = visualize_training_progress(training_metrics, output_dir=plot_dir)
        print(f"Training progress plot saved to {training_plot_path}")
    else:
        print("Warning: No training rewards collected, skipping training visualization")
    
    # Evaluate the trained agent
    print("Evaluating trained agent...")
    eval_episodes = 10
    eval_metrics = evaluate_agent(env, agent, num_episodes=eval_episodes)
    
    # Save evaluation metrics to text file for reference
    eval_metrics_path = f"{run_dir}/evaluation_metrics.txt"
    with open(eval_metrics_path, "w") as f:
        f.write(f"Average Reward: {eval_metrics['avg_reward']:.4f}\n")
        f.write(f"Success Rate: {eval_metrics['success_rate']:.4f}\n")
        f.write(f"Action Distribution: Conservative: {eval_metrics['action_distribution'].get(0, 0):.2f}, " + 
                f"Aggressive: {eval_metrics['action_distribution'].get(1, 0):.2f}\n")
        f.write(f"Average Conservative Reward: {eval_metrics['avg_conservative_reward']:.4f}\n")
        f.write(f"Average Aggressive Reward: {eval_metrics['avg_aggressive_reward']:.4f}\n")
    
    # Create evaluation visualizations
    print("Generating evaluation visualization...")
    eval_plot_path = visualize_evaluation(eval_metrics, feature_columns, output_dir=plot_dir)
    print(f"Evaluation plot saved to {eval_plot_path}")
    
    # Save trained model
    model_path = f"{run_dir}/ad_optimization_model.pt"
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'feature_columns': feature_columns,
        'training_metrics': training_metrics
    }, model_path)
    print(f"Model saved to {model_path}")
    
    print(f"Pipeline completed successfully. All results saved to {run_dir}")
    return agent, training_metrics, eval_metrics

# Script entry point
if __name__ == "__main__":
    main()

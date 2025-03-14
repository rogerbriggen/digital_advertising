#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from typing import Dict, Optional, Any, Tuple
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data import OneHot, Bounded, Unbounded, Binary, Composite
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate
# Define the file path
file_path = 'data/organized_dataset.csv'

# Generate Realistic Synthetic Data. This is coming from Ilja's code
# Platzierung:
#    - Organisch: Erscheint aufgrund des Suchalgorithmus, ohne Bezahlung.
#    - Paid: Wird aufgrund einer Werbekampagne oder bezahlten Platzierung angezeigt.
# Kosten:
#    - Organisch: Es fallen in der Regel keine direkten Kosten pro Klick oder Impression an.
#    - Paid: Werbetreibende zahlen oft pro Klick (CPC) oder pro Impression (CPM = pro Sichtkontakt, unabhängig ob jemand klickt oder nicht).
def generate_synthetic_data(num_samples=1000):
    data = {
        "keyword": [f"Keyword_{i}" for i in range(num_samples)],        # Eindeutiger Name oder Identifier für das Keyword
        "competitiveness": np.random.uniform(0, 1, num_samples),        # Wettbewerbsfähigkeit des Keywords (Wert zwischen 0 und 1). Je mehr Leute das Keyword wollen, desto näher bei 1 und somit desto teurer.
        "difficulty_score": np.random.uniform(0, 1, num_samples),       # Schwierigkeitsgrad des Keywords organisch gute Platzierung zu erreichen (Wert zwischen 0 und 1). 1 = mehr Aufwand und Optimierung nötig.
        "organic_rank": np.random.randint(1, 11, num_samples),          # Organischer Rang, z.B. Position in Suchergebnissen (1 bis 10)
        "organic_clicks": np.random.randint(50, 5000, num_samples),     # Anzahl der Klicks auf organische Suchergebnisse
        "organic_ctr": np.random.uniform(0.01, 0.3, num_samples),       # Klickrate (CTR) für organische Suchergebnisse
        "paid_clicks": np.random.randint(10, 3000, num_samples),        # Anzahl der Klicks auf bezahlte Anzeigen
        "paid_ctr": np.random.uniform(0.01, 0.25, num_samples),         # Klickrate (CTR) für bezahlte Anzeigen
        "ad_spend": np.random.uniform(10, 10000, num_samples),          # Werbebudget bzw. Ausgaben für Anzeigen
        "ad_conversions": np.random.randint(0, 500, num_samples),       # Anzahl der Conversions (Erfolge) von Anzeigen
        "ad_roas": np.random.uniform(0.5, 5, num_samples),              # Return on Ad Spend (ROAS) für Anzeigen, wobei Werte < 1 Verlust anzeigen
        "conversion_rate": np.random.uniform(0.01, 0.3, num_samples),   # Conversion-Rate (Prozentsatz der Besucher, die konvertieren)
        "cost_per_click": np.random.uniform(0.1, 10, num_samples),      # Kosten pro Klick (CPC)
        "cost_per_acquisition": np.random.uniform(5, 500, num_samples), # Kosten pro Akquisition (CPA)
        "previous_recommendation": np.random.choice([0, 1], size=num_samples),  # Frühere Empfehlung (0 = nein, 1 = ja)
        "impression_share": np.random.uniform(0.1, 1.0, num_samples),   # Anteil an Impressionen (Sichtbarkeit der Anzeige) im Vergleich mit allen anderen die dieses Keyword wollen
        "conversion_value": np.random.uniform(0, 10000, num_samples)    # Monetärer Wert der Conversions (Ein monetärer Wert, der den finanziellen Nutzen aus den erzielten Conversions widerspiegelt. Dieser Wert gibt an, wie viel Umsatz oder Gewinn durch die Conversions generiert wurde – je höher der Wert, desto wertvoller sind die Conversions aus Marketingsicht.)
    }
    return pd.DataFrame(data)

# Example
'''
test = generate_synthetic_data(10)
test.head()
print(test.shape)
print(test.columns)
'''


# Load synthetic dataset
''''
dataset = generate_synthetic_data(1000)
'''


def read_and_organize_csv(file_path):
    """
    Reads a CSV file, organizes the data by keywords, and returns the organized DataFrame.
    This function performs the following steps:
    1. Reads the CSV file from the given file path into a DataFrame.
    2. Drops the 'step' column from the DataFrame.
    3. Extracts unique keywords from the 'keyword' column.
    4. Organizes the data by iterating through the first 5000 rows for each keyword and concatenates the rows into a new DataFrame.
    Args:
        file_path (str): The file path to the CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the organized data, with the index reset.
    """
    df = pd.read_csv(file_path)
    organized_data = pd.DataFrame()

    # Skip the 'step' column
    df = df.drop(columns=['step'])

    # Get unique keywords
    keywords = df['keyword'].unique()

    # Organize data
    for i in range(5000):
        for keyword in keywords:
            keyword_data = df[df['keyword'] == keyword]
            if len(keyword_data) > i:
                organized_data = pd.concat([organized_data, keyword_data.iloc[[i]]])

    return organized_data.reset_index(drop=True)

# Example usage
''''
dataset = pd.read_csv('data/organized_dataset.csv')
dataset.head()
'''

def split_dataset_by_ratio(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and test sets based on keywords.
    
    Args:
        dataset (pd.DataFrame): The dataset to split.
        train_ratio (float): Ratio of keywords to include in the training set (0.0-1.0).
        
    Returns:
        tuple: (training_dataset, test_dataset)
    """
    # Get all unique keywords
    keywords = dataset['keyword'].unique()
    
    # Fetch the amount of rows for each keyword
    entries_in_dataset = len(dataset) / keywords.size
    
    # Split rows into training and test sets
    rows_training = round((len(dataset) * train_ratio) / keywords.size) * keywords.size # Round to the nearest multiple of the number of keywords
    rows_test = int(len(dataset) - rows_training)

    # Create training and test datasets
    train_dataset = dataset.iloc[0:rows_training].reset_index(drop=True)
    test_dataset = dataset.iloc[rows_training:].reset_index(drop=True)
    
    print(f"Training dataset: {len(train_dataset)} rows, {len(train_dataset['keyword'].unique())} keywords")
    print(f"Test dataset: {len(test_dataset)} rows, {len(test_dataset['keyword'].unique())} keywords")
    
    return train_dataset, test_dataset


def get_entry_from_dataset(df, index):
    """
    Retrieves a subset of rows from the DataFrame based on unique keywords.
    This function calculates the number of unique keywords in the DataFrame
    and uses this number to determine the subset of rows to return. The subset
    is determined by the given index and the number of unique keywords.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the dataset.
    index (int): The index to determine which subset of rows to retrieve.
    Returns:
    pandas.DataFrame: A subset of the DataFrame containing rows corresponding
                      to the specified index and the number of unique keywords.
    """
    # Count unique keywords
    seen_keywords = set()
    if not hasattr(get_entry_from_dataset, "unique_keywords"):
        seen_keywords = set()
        for i, row in df.iterrows():
            keyword = row['keyword']
            if keyword in seen_keywords:
                break
            seen_keywords.add(keyword)
        get_entry_from_dataset.unique_keywords = seen_keywords
        get_entry_from_dataset.keywords_amount = len(seen_keywords)
    else:
        seen_keywords = get_entry_from_dataset.unique_keywords

    # Get the subset of rows based on the index
    keywords_amount = get_entry_from_dataset.keywords_amount
    return df.iloc[index * keywords_amount:index * keywords_amount + keywords_amount].reset_index(drop=True)

# Example usage
'''
entry = get_entry_from_dataset(dataset, 0)
print(entry)

entry = get_entry_from_dataset(dataset, 1)
print(entry)
'''


# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    """
    AdOptimizationEnv is an environment for optimizing digital advertising strategies using reinforcement learning.
    Attributes:
        initial_cash (float): Initial cash balance for the environment.
        dataset (pd.DataFrame): Dataset containing keyword metrics.
        num_features (int): Number of features for each keyword.
        num_keywords (int): Number of keywords in the dataset.
        action_spec (OneHot): Action specification for the environment.
        reward_spec (Unbounded): Reward specification for the environment.
        observation_spec (Composite): Observation specification for the environment.
        done_spec (Composite): Done specification for the environment.
        current_step (int): Current step in the environment.
        holdings (torch.Tensor): Tensor representing the current holdings of keywords.
        cash (float): Current cash balance.
        obs (TensorDict): Current observation of the environment.
    Methods:
        __init__(self, dataset, initial_cash=100000.0, device="cpu"):
            Initializes the AdOptimizationEnv with the given dataset, initial cash, and device.
        _reset(self, tensordict=None):
            Resets the environment to the initial state and returns the initial observation.
        _step(self, tensordict):
            Takes a step in the environment using the given action and returns the next state, reward, and done flag.
        _compute_reward(self, action, current_pki, action_idx):
            Computes the reward based on the selected keyword's metrics.
        _set_seed(self, seed: Optional[int]):
            Sets the random seed for the environment.
    """

    def __init__(self, dataset, initial_cash=100000.0, device="cpu"):
        """
        Initializes the digital advertising environment.
        Args:
            dataset (Any): The dataset containing keyword features and other relevant data.
            initial_cash (float, optional): The initial amount of cash available for advertising. Defaults to 100000.0.
            device (str, optional): The device to run the environment on, either "cpu" or "cuda". Defaults to "cpu".
        Attributes:
            initial_cash (float): The initial amount of cash available for advertising.
            dataset (Any): The dataset containing keyword features and other relevant data.
            num_features (int): The number of features in the dataset.
            num_keywords (int): The number of keywords in the dataset.
            action_spec (OneHot): The specification for the action space, which includes selecting a keyword to buy or choosing to buy nothing.
            reward_spec (Unbounded): The specification for the reward space, which is unbounded and of type torch.float32.
            observation_spec (Composite): The specification for the observation space, which includes keyword features, cash, holdings, and step count.
            done_spec (Composite): The specification for the done space, which includes flags for done, terminated, and truncated states.
        """
        super().__init__(device=device)
        self.initial_cash = initial_cash
        self.dataset = dataset
        self.num_features = len(feature_columns)
        self.num_keywords = get_entry_from_dataset(self.dataset, 0).shape[0]
        self.action_spec = OneHot(n=self.num_keywords + 1) # select which one to buy or the last one to buy nothing
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.observation_spec = Composite(
            observation = Composite(
                keyword_features=Unbounded(shape=(self.num_keywords, self.num_features), dtype=torch.float32),
                cash=Unbounded(shape=(1,), dtype=torch.float32),
                holdings=Bounded(low=0, high=1, shape=(self.num_keywords,), dtype=torch.int, domain="discrete")
            ),
            step_count=Unbounded(shape=(1,), dtype=torch.int64)
        )
        self.done_spec = Composite(
            done=Binary(shape=(1,), dtype=torch.bool),
            terminated=Binary(shape=(1,), dtype=torch.bool),
            truncated=Binary(shape=(1,), dtype=torch.bool)
        )

        self.feature_means = torch.tensor(dataset[feature_columns].mean().values,
                                          dtype=torch.float32, device=device)
        self.feature_stds = torch.tensor(dataset[feature_columns].std().values,
                                         dtype=torch.float32, device=device)
        # Prevent division by zero
        self.feature_stds = torch.where(self.feature_stds > 0, self.feature_stds,
                                        torch.ones_like(self.feature_stds))

        # Add cash normalization
        self.cash_mean = initial_cash / 2
        self.cash_std = initial_cash / 4

        self.reset()

    def _reset(self, tensordict: TensorDict =None):
        """
        Resets the environment to its initial state.
        Args:
            tensordict (TensorDict, optional): A TensorDict to be updated with the reset state. If None, a new TensorDict is created.
        Returns:
            TensorDict: A TensorDict containing the reset state of the environment, including:
                - "done" (torch.tensor): A boolean tensor indicating if the episode is done.
                - "observation" (TensorDict): A TensorDict containing the initial observation with:
                    - "keyword_features" (torch.tensor): Features of the current keywords.
                    - "cash" (torch.tensor): The initial cash balance.
                    - "holdings" (torch.tensor): The initial holdings state for each keyword.
                - "step_count" (torch.tensor): The current step count, initialized to 0.
                - "terminated" (torch.tensor): A boolean tensor indicating if the episode is terminated.
                - "truncated" (torch.tensor): A boolean tensor indicating if the episode is truncated.
        """
        self.current_step = 0
        self.holdings = torch.zeros(self.num_keywords, dtype=torch.int, device=self.device) # 0 = not holding, 1 = holding keyword
        self.cash = self.initial_cash
        #sample = self.dataset.sample(1)
        #state = torch.tensor(sample[feature_columns].values, dtype=torch.float32).squeeze()
        # Create the initial observation.
        keyword_features = torch.tensor(get_entry_from_dataset(self.dataset, self.current_step)[feature_columns].values, dtype=torch.float32, device=self.device)
        keyword_features = (keyword_features - self.feature_means) / self.feature_stds
        cash_normalized = (torch.tensor(self.cash, dtype=torch.float32,
                                        device=self.device) - self.cash_mean) / self.cash_std
        obs = TensorDict({
            "keyword_features": keyword_features,  # Current pki for each keyword
            "cash": torch.tensor(cash_normalized.clone().detach(), dtype=torch.float32, device=self.device),  # Current cash balance
            "holdings": self.holdings.clone()  # 1 for each keyword if we are holding
        }, batch_size=[])
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=[])
        else:
            tensordict = tensordict.empty()
        tensordict = tensordict.update({
            "done": torch.tensor(False, dtype=torch.bool, device=self.device),
            "observation": obs,
            "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
            "terminated": torch.tensor(False, dtype=torch.bool, device=self.device),
            "truncated": torch.tensor(False, dtype=torch.bool, device=self.device)
        })
        
        self.obs = obs
        # print(f'Reset: Step: {self.current_step}')
        return tensordict


    def _step(self, tensordict: TensorDict):
        """
        Perform a single step in the environment using the provided tensor dictionary.
        Args:
            tensordict (TensorDict): A dictionary containing the current state and action.
        Returns:
            TensorDict: A dictionary containing the next state, reward, and termination status.
        The function performs the following steps:
        1. Extracts the action from the input tensor dictionary.
        2. Determines the index of the selected keyword.
        3. Retrieves the current entry from the dataset based on the current step.
        4. Updates the holdings based on the selected action.
        5. Calculates the reward based on the action taken.
        6. Advances to the next time step and checks for termination conditions.
        7. Retrieves the next keyword features for the subsequent state.
        8. Updates the observation state with the new keyword features, cash balance, and holdings.
        9. Updates the tensor dictionary with the new state, reward, and termination status.
        10. Returns the updated tensor dictionary containing the next state, reward, and termination status.
        """
        # Get the action from the input tensor dictionary. 
        action = tensordict["action"]
        #action_idx = action.argmax(dim=-1).item()  # Get the index of the selected keyword
        true_indices = torch.nonzero(action, as_tuple=True)[0]
        action_idx = true_indices[0] if len(true_indices) > 0 else self.action_spec.n - 1

        current_pki = get_entry_from_dataset(self.dataset, self.current_step)
        #action = tensordict["action"].argmax(dim=-1).item()


        # Update cash based on the action
        ad_roas = 0.0
        if action_idx < self.num_keywords:
            # Get the selected keyword's ad spend
            selected_keyword = current_pki.iloc[action_idx.item()]
            ad_cost = selected_keyword["ad_spend"]
            ad_revenue = selected_keyword["conversion_value"]
            ad_roas = selected_keyword["ad_roas"]

            # we assume the marketing budget is 10% of the cash
            if (self.cash * 0.1) >= ad_cost:
                # When enough balance, update cash with ad revenue and deduct ad cost
                self.cash -= ad_cost
                self.cash += ad_revenue

        # Update holdings based on action (only one keyword is selected)
        new_holdings = torch.zeros_like(self.holdings)
        if action_idx < self.num_keywords:
            new_holdings[action_idx] = 1
        self.holdings = new_holdings

        # Calculate the reward based on the action taken.
        reward = self._compute_reward(action, current_pki, action_idx, ad_roas)

         # Move to the next time step.
        self.current_step += 1
        terminated = self.cash < 0 or self.current_step >= (len(self.dataset) // self.num_keywords) - 2 # -2 to avoid going over the last index
        truncated = False

        # Get next pki for the keywords
        next_keyword_features = torch.tensor(get_entry_from_dataset(self.dataset, self.current_step)[feature_columns].values, dtype=torch.float32, device=self.device)
        next_keyword_features = (next_keyword_features - self.feature_means) / self.feature_stds
        # todo: most probably we need to remove some columns from the state so we only have the features for the agent to see... change it also in reset
        cash_normalized = (torch.tensor(self.cash, dtype=torch.float32,
                                        device=self.device) - self.cash_mean) / self.cash_std
        next_obs = TensorDict({
            "keyword_features": next_keyword_features,  # next pki for each keyword
            "cash": cash_normalized.clone().detach(),  # Current cash balance
            "holdings": self.holdings.clone()
        }, batch_size=[])
        
        # Update the state
        self.obs = next_obs
        print(f'Step: {self.current_step}, Action: {action_idx}, Reward: {reward}, Cash: {self.cash}')

        # PK: todo: is this really needed? seems tensordict is not used anymore after this assignment
        tensordict["done"] = torch.as_tensor(bool(terminated or truncated), dtype=torch.bool, device=self.device)
        tensordict["observation"] = self.obs
        tensordict["reward"] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        tensordict["step_count"] = torch.tensor(self.current_step-1, dtype=torch.int64, device=self.device)
        tensordict["terminated"] = torch.tensor(bool(terminated), dtype=torch.bool, device=self.device)
        tensordict["truncated"] = torch.tensor(bool(truncated), dtype=torch.bool, device=self.device)
        next = TensorDict({
            "done": torch.tensor(bool(terminated or truncated), dtype=torch.bool, device=self.device),
            "observation": next_obs,
            "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
            "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
            "terminated": torch.tensor(bool(terminated), dtype=torch.bool, device=self.device),
            "truncated": torch.tensor(bool(truncated), dtype=torch.bool, device=self.device)

        }, batch_size=tensordict.batch_size)
        
        return next


    def _compute_reward(self, action, current_pki, action_idx, ad_roas):
        """Compute reward based on the selected keyword's metrics"""
        adjusted_reward = 0 if action_idx < self.num_keywords else 1
        if ad_roas > 0:
            adjusted_reward = np.log(ad_roas)
        missing_rewards = []
        # Iterate through all keywords
        for i in range(self.num_keywords):
            sample = current_pki.iloc[i]
            if action[i] == False:
                missing_rewards.append(sample["ad_roas"])
        # Adjust reward based on missing rewards to penalize the agent when not selecting keywords with high(er) ROAS
        # clipping reduces the variance of the rewards
        return np.clip(adjusted_reward - np.mean(missing_rewards) * 0.2, -2, 2)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng


class FlattenInputs(nn.Module):
    """
    A custom PyTorch module to flatten and combine keyword features, cash, and holdings into a single tensor.
    Methods
    -------
    forward(keyword_features, cash, holdings)
        Flattens and combines the input tensors into a single tensor.
    Parameters
    ----------
    keyword_features : torch.Tensor
        A tensor containing keyword features with shape [batch, num_keywords, feature_dim] or [num_keywords, feature_dim].
    cash : torch.Tensor
        A tensor containing cash values with shape [batch] or [batch, 1] or a scalar.
    holdings : torch.Tensor
        A tensor containing holdings with shape [batch, num_keywords] or [num_keywords].
    Returns
    -------
    torch.Tensor
        A combined tensor with all inputs flattened and concatenated along the appropriate dimension.
    """
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


class ModelHandler:
    """
    A class to handle saving and loading of models for the digital advertising system.
    
    This class provides functionality to:
    1. Save models during training based on performance criteria
    2. Load models for inference or continued training
    3. Manage model versioning and metadata
    """
    
    def __init__(self, save_dir: str = 'saves'):
        """
        Initialize the ModelHandler.
        
        Args:
            save_dir (str): Directory to save models to and load models from.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, 
                  policy: TensorDictSequential,
                  optim: Optional[torch.optim.Optimizer] = None,
                  metadata: Dict[str, Any] = None,
                  filename: Optional[str] = None) -> str:
        """
        Save the model and related data.
        
        Args:
            policy: The policy model to save
            optim: Optional optimizer to save for continued training
            metadata: Additional information to save with the model
            filename: Custom filename, if None generates a timestamped name
            
        Returns:
            str: Path to the saved model file
        """
        if metadata is None:
            metadata = {}
            
        # Create a save dictionary with the policy
        save_dict = {
            'policy_state_dict': policy.state_dict(),
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        # Add optimizer if provided
        if optim is not None:
            save_dict['optimizer_state_dict'] = optim.state_dict()
            
        # Generate filename if not provided
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            reward = metadata.get('test_reward', 0)
            steps = metadata.get('total_steps', 0)
            filename = f"model_{timestamp}_reward{reward:.2f}_steps{steps}.pt"
        
        # Ensure file has .pt extension
        if not filename.endswith('.pt'):
            filename += '.pt'
            
        filepath = os.path.join(self.save_dir, filename)
        
        # Save the model
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, 
                  policy: TensorDictSequential,
                  filepath: str, 
                  device: torch.device,
                  optim: Optional[torch.optim.Optimizer] = None,
                  inference_only: bool = False) -> Tuple[TensorDictSequential, Dict[str, Any]]:
        """
        Load a model from a file.
        
        Args:
            policy: The policy model architecture to load weights into
            filepath: Path to the model file
            device: Device to load the model to
            optim: Optional optimizer to load state into
            inference_only: If True, sets model to eval mode and doesn't load optimizer
            
        Returns:
            Tuple: (loaded_policy, metadata_dict)
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load the policy state dict
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Set to evaluation mode if inference only
        if inference_only:
            policy.eval()
        
        # Load optimizer if provided and available in checkpoint
        if optim is not None and not inference_only and 'optimizer_state_dict' in checkpoint:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Get metadata
        metadata = checkpoint.get('metadata', {})
        
        print(f"Model loaded from {filepath}")
        if 'test_reward' in metadata:
            print(f"Test reward: {metadata['test_reward']}")
        if 'total_steps' in metadata:
            print(f"Training steps: {metadata['total_steps']}")
            
        return policy, metadata
    
    def find_best_model(self) -> Optional[str]:
        """
        Find the best performing model in the save directory.
        
        Returns:
            str or None: Path to the best model file, or None if no models found
        """
        best_reward = float('-inf')
        best_model_path = None
        
        for filename in os.listdir(self.save_dir):
            if filename.endswith('.pt'):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    checkpoint = torch.load(filepath, map_location='cpu')
                    metadata = checkpoint.get('metadata', {})
                    reward = metadata.get('test_reward', float('-inf'))
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_model_path = filepath
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        if best_model_path:
            print(f"Found best model: {best_model_path} with reward: {best_reward}")
        else:
            print("No valid models found.")
            
        return best_model_path


def create_policy(env, feature_dim, num_keywords, device):
    """
    Creates a policy network with the standard architecture.
    
    Args:
        env: Environment containing action_spec
        feature_dim: Dimension of features per keyword
        num_keywords: Number of keywords
        device: Device to create the policy on
        
    Returns:
        policy: The complete policy model
    """
    action_dim = env.action_spec.shape[-1]
    total_input_dim = feature_dim * num_keywords + 1 + num_keywords  # features per keyword + cash + holdings
    
    # Create the flattening module
    flatten_module = TensorDictModule(
        FlattenInputs(),
        in_keys=[("observation", "keyword_features"), ("observation", "cash"), ("observation", "holdings")],
        out_keys=["flattened_input"]
    )
    
    # Create the value network
    value_mlp = MLP(
        in_features=total_input_dim, 
        out_features=action_dim, 
        num_cells=[256, 256, 128, 64],  # Deeper and wider architecture
        activation_class=nn.ReLU  # ReLU often performs better than Tanh
    )
    
    value_net = TensorDictModule(value_mlp, in_keys=["flattened_input"], out_keys=["action_value"])
    
    # Combine into the complete policy
    policy = TensorDictSequential(flatten_module, value_net, QValueModule(spec=env.action_spec))
    
    return policy.to(device)

def run_inference(model_path, dataset_test, device, feature_columns):
    """
    Run inference using a saved model
    
    Args:
        model_path: Path to the saved model
        dataset_test: Test dataset 
        device: Device to run on
        feature_columns: List of feature column names
    """
    # Create test environment
    test_env = AdOptimizationEnv(dataset_test, device=device)
    
    # Get dimensions
    feature_dim = len(feature_columns)
    num_keywords = test_env.num_keywords
    
    # Create a fresh policy with the same architecture
    inference_policy = create_policy(test_env, feature_dim, num_keywords, device)
    
    # Load the saved model handler
    model_handler = ModelHandler()
    inference_policy, metadata = model_handler.load_model(
        policy=inference_policy,
        filepath=model_path,
        device=device,
        inference_only=True
    )
    
    # Run inference
    test_td = test_env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        with torch.no_grad():
            test_td = inference_policy(test_td)
        test_td = test_env.step(test_td)
        reward = test_td["reward"].item()
        total_reward += reward
        done = test_td["done"].item()
        
        print(f"Step: {test_td['step_count'].item()}, Action: {test_td['action'].argmax().item()}, Reward: {reward}")
    
    print(f"Total inference reward: {total_reward}")
    return total_reward, inference_policy

#def learn():
    # Load the organized dataset
    # dataset = pd.read_csv('data/organized_dataset.csv')
    # Split it into training and test data
    # dataset_training, dataset_test = split_dataset_by_ratio(dataset, train_ratio=0.8)
def learn(params=None, train_data=None, test_data=None):
    # Load the organized dataset
    # Check if the file exists
    if os.path.exists(file_path):
        # If file exists, load it directly
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded from {file_path}")
    else:
        # If file doesn't exist, generate synthetic data
        print(f"File {file_path} not found. Generating synthetic data...")       
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataset = generate_synthetic_data(1000)
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded from newly created {file_path}")
    #    # Split it into training and test data
    #    dataset_training, dataset_test = split_dataset_by_ratio(dataset, train_ratio=0.8)
    # dataset = generate_synthetic_data(1000)
    dataset_training, dataset_test = split_dataset_by_ratio(dataset, train_ratio=0.8)
    train_data, test_data = dataset_training, dataset_test

    # Initialize Environment
    env = AdOptimizationEnv(dataset_training, device=device)
    
    # Define data and dimensions
    feature_dim = len(feature_columns)
    num_keywords = env.num_keywords

    # Create the main policy for training
    policy = create_policy(env, feature_dim, num_keywords, device)

    # Create the evaluation policy (now using the same architecture)
    policy_eval = create_policy(env, feature_dim, num_keywords, device)

    exploration_module = EGreedyModule(
        env.action_spec, annealing_num_steps=100_000, eps_init=0.9, eps_end=0.01
    )
    exploration_module = exploration_module.to(device)
    policy_explore = TensorDictSequential(policy, exploration_module).to(device)

    init_rand_steps = 5000
    frames_per_batch = 100
    optim_steps = 10
    collector = SyncDataCollector(
        env,
        policy_explore,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        init_random_frames=init_rand_steps,
    )
    replay_buffer_size = 100_000
    rb = ReplayBuffer(storage=LazyTensorStorage(replay_buffer_size))

    loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True).to(device)
#    lr=0.001
#    weight_decay=1e-5
#    eps=0.99
    if params is None:
        params = {
            'lr': 0.001,
            'batch_size': 128,
            'gamma': 0.99,
            'weight_decay': 1e-5,
            'eps_init': 0.99,
            'eps_end': 0.01
        }

    # Extract hyperparameters
    lr = params.get('lr', 0.001)
    batch_size = params.get('batch_size', 128)
    gamma = params.get('gamma', 0.99)
    weight_decay = params.get('weight_decay', 1e-5)
    eps = params.get('eps_init', 0.99)
    eps_end = params.get('eps_end', 0.01)

    optim = Adam(loss.parameters(), lr=lr, weight_decay=weight_decay)  # Add weight decay for regularization
    updater = SoftUpdate(loss, eps=eps)

    total_count = 0
    total_episodes = 0
    t0 = time.time()
    # Evaluation parameters
    evaluation_frequency = 1000  # Run evaluation every 1000 steps
    best_test_reward = float('-inf')
    test_env = AdOptimizationEnv(dataset_test, device=device)  # Create a test environment with the test dataset
    model_handler = ModelHandler(save_dir='saves')
 
    # Tensorboard vorbereiten
    writer = SummaryWriter()
    # Write the hyperparameters to tensorboard
    writer.add_text("Feature Columns", str(feature_columns))
    writer.add_text("Num Keywords", str(num_keywords))
    writer.add_text("init_rand_steps", str(init_rand_steps))  
    writer.add_text("frames_per_batch", str(frames_per_batch))
    writer.add_text("optim_steps", str(optim_steps))
    writer.add_text("lr", str(lr))
    writer.add_text("weight_decay", str(weight_decay))
    writer.add_text("eps", str(eps))
 
    for i, data in enumerate(collector):
        # Write data in replay buffer
        step_count = data["step_count"]

        print(f'data: step_count: {step_count}')
        rb.extend(data.to(device))
        max_length = rb[:]["step_count"].max()
        if len(rb) > init_rand_steps:
            # Optim loop (we do several optim steps per batch collected for efficiency)
            for _ in range(optim_steps):
                sample = rb.sample(128)
                total_count += data.numel()

                # Make sure sample is on the correct device
                sample = sample.to(device)  # Move the sample to the specified device
                loss_vals = loss(sample)
                writer.add_scalar("Loss Value", loss_vals["loss"].item(), total_count)
                loss_vals["loss"].backward()
                optim.step()
                optim.zero_grad()
                # Update exploration factor
                exploration_module.step(data.numel())
                # Update target params
                updater.step()
                if i % 10 == 0:  # Fixed condition (was missing '== 0')
                    print(f"Max num steps: {max_length}, rb length {len(rb)}")

                total_episodes += data["next", "done"].sum()

                # Evaluate on test data periodically
                if total_count % evaluation_frequency == 0:
                    print(f"\n--- Testing model performance after {total_count} training steps ---")
                    # Use policy without exploration for evaluation
                    policy_eval.load_state_dict(policy.state_dict())  # Just use the trained policy without exploration
                    policy_eval.eval()

                    # Reset the test environment
                    test_td = test_env.reset()
                    total_test_reward = 0.0
                    done = False
                    max_test_steps = 100  # Limit test steps to avoid infinite loops
                    test_step = 0

                    # Run the model on test environment until done or max steps reached
                    while not done and test_step < max_test_steps:
                        # Forward pass through policy without exploration
                        with torch.no_grad():
                            # Get Q-values
                            test_td = policy_eval(test_td)

                            # Check and handle NaN values in Q-values
                            q_values = test_td["action_value"]
                            best_idx = q_values.argmax(dim=-1).item()

                            # Create one-hot action
                            action = torch.zeros_like(q_values)
                            action[..., best_idx] = 1
                            test_td["action"] = action

                        # Step in the test environment
                        test_td = test_env.step(test_td)
                        reward = test_td["reward"].item()
                        total_test_reward += reward
                        done = test_td["done"].item()
                        test_step += 1

                    writer.add_scalar("Test performance", total_test_reward, total_count)
                    print(f"Test performance: Total reward = {total_test_reward}, Steps = {test_step}")

                    # Save model if it's the best so far
                    if total_test_reward > best_test_reward:
                        best_test_reward = total_test_reward
                        print(f"New best model! Saving with reward: {best_test_reward}")

                        # Save the model
                        model_handler.save_model(
                            policy=policy,
                            optim=optim,
                            metadata={
                                'total_steps': total_count,
                                'test_reward': best_test_reward,
                                'test_steps': test_step,
                                'num_keywords': num_keywords,
                                'feature_columns': feature_columns
                            },
                            filename=f"best_model.pt"  # Overwrite the same file for best model
                        )
                        print(policy.state_dict())

                    print("--- Testing completed ---\n")

        if total_count > 10_000:
            break

    t1 = time.time()

    print(
        f"Finished after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
    )
    print(f"Best test performance: {best_test_reward}")

    # Run inference with the best model
    best_model_path = model_handler.find_best_model()
    if best_model_path:
        total_reward, _ = run_inference(best_model_path, dataset_test, device, feature_columns)
        return total_reward
    else:
        return best_test_reward

#some global variables
# Select the best device for our machine
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(device)

# Define the feature columns
feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", "ad_roas", "conversion_rate", "cost_per_click"]

if __name__ == "__main__":
    learn()


''''
Todo:
- ✅ Clean up the code
- ✅ Split training and test data (RB)
- ✅ Implement tensorboard (PK, MAC)
- Implement the visualization (see tensorboard) (EO)
- ✅ Implement the saving of the model (RB)
- ✅Implement the inference (RB)
- Implement the optuna hyperparameter tuning (UT)
'''''

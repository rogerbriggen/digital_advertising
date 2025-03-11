#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from torchrl.envs import EnvBase
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.objectives import DQNLoss
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from typing import Optional
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.data import OneHot, Bounded, Unbounded, Binary, MultiCategorical, Composite, UnboundedContinuous


from tensordict.nn import TensorDictModule, TensorDictSequential


# Specify device explicitly
device = torch.device("cpu")  # or "cuda" if you have GPU support

# Generate Realistic Synthetic Data
#Platzierung:
#Organisch: Erscheint aufgrund des Suchalgorithmus, ohne Bezahlung.
#Paid: Wird aufgrund einer Werbekampagne oder bezahlten Platzierung angezeigt.
#Kosten:
#Organisch: Es fallen in der Regel keine direkten Kosten pro Klick oder Impression an.
#Paid: Werbetreibende zahlen oft pro Klick (CPC) oder pro Impression (CPM = pro Sichtkontakt, unabhängig ob jemand klickt oder nicht).
def generate_synthetic_data(num_samples=1000):
    data = {
        "keyword": [f"Keyword_{i}" for i in range(num_samples)],  # Eindeutiger Name oder Identifier für das Keyword
        "competitiveness": np.random.uniform(0, 1, num_samples),     # Wettbewerbsfähigkeit des Keywords (Wert zwischen 0 und 1). Je mehr Leute das Keyword wollen, desto näher bei 1 und somit desto teurer.
        "difficulty_score": np.random.uniform(0, 1, num_samples),      # Schwierigkeitsgrad des Keywords organisch gute Platzierung zu erreichen (Wert zwischen 0 und 1). 1 = mehr Aufwand und Optimierung nötig.
        "organic_rank": np.random.randint(1, 11, num_samples),         # Organischer Rang, z.B. Position in Suchergebnissen (1 bis 10)
        "organic_clicks": np.random.randint(50, 5000, num_samples),    # Anzahl der Klicks auf organische Suchergebnisse
        "organic_ctr": np.random.uniform(0.01, 0.3, num_samples),      # Klickrate (CTR) für organische Suchergebnisse
        "paid_clicks": np.random.randint(10, 3000, num_samples),       # Anzahl der Klicks auf bezahlte Anzeigen
        "paid_ctr": np.random.uniform(0.01, 0.25, num_samples),        # Klickrate (CTR) für bezahlte Anzeigen
        "ad_spend": np.random.uniform(10, 10000, num_samples),         # Werbebudget bzw. Ausgaben für Anzeigen
        "ad_conversions": np.random.randint(0, 500, num_samples),      # Anzahl der Conversions (Erfolge) von Anzeigen
        "ad_roas": np.random.uniform(0.5, 5, num_samples),             # Return on Ad Spend (ROAS) für Anzeigen, wobei Werte < 1 Verlust anzeigen
        "conversion_rate": np.random.uniform(0.01, 0.3, num_samples),    # Conversion-Rate (Prozentsatz der Besucher, die konvertieren)
        "cost_per_click": np.random.uniform(0.1, 10, num_samples),     # Kosten pro Klick (CPC)
        "cost_per_acquisition": np.random.uniform(5, 500, num_samples),  # Kosten pro Akquisition (CPA)
        "previous_recommendation": np.random.choice([0, 1], size=num_samples),  # Frühere Empfehlung (0 = nein, 1 = ja)
        "impression_share": np.random.uniform(0.1, 1.0, num_samples),  # Anteil an Impressionen (Sichtbarkeit der Anzeige) im Vergleich mit allen anderen die dieses Keyword wollen
        "conversion_value": np.random.uniform(0, 10000, num_samples)   # Monetärer Wert der Conversions (Ein monetärer Wert, der den finanziellen Nutzen aus den erzielten Conversions widerspiegelt. Dieser Wert gibt an, wie viel Umsatz oder Gewinn durch die Conversions generiert wurde – je höher der Wert, desto wertvoller sind die Conversions aus Marketingsicht.)
    }
    return pd.DataFrame(data)

test = generate_synthetic_data(10)
test.head()
print(test.shape)
print(test.columns)


def getKeywords():
    return ["investments", "stocks", "crypto", "cryptocurrency", "bitcoin", "real estate", "gold", "bonds", "broker", "finance", "trading", "forex", "etf", "investment fund", "investment strategy", "investment advice", "investment portfolio", "investment opportunities", "investment options", "investment calculator", "investment plan", "investment account", "investment return", "investment risk", "investment income", "investment growth", "investment loss", "investment profit", "investment return calculator", "investment return formula", "investment return rate"]


def generateData():
    seed = 42  # or any integer of your choice
    random.seed(seed)      # Sets the seed for the Python random module
    np.random.seed(seed)   # Sets the seed for NumPy's random generator
    torch.manual_seed(seed)  # Sets the seed for PyTorch

    # If you're using CUDA as well, you may also set:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Generate synthetic data
    # Do it 1000 times
    dataset = pd.DataFrame()
    for i in range(1000):
        # append to dataset
        dataset = generate_synthetic_data(len(getKeywords()))
        


# Load synthetic dataset
dataset = generate_synthetic_data(1000)
feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", "ad_roas", "conversion_rate", "cost_per_click"]


def read_and_organize_csv(file_path):
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
#organized_dataset = read_and_organize_csv('18 TorchRL Ads/balanced_ad_dataset_real_keywords.csv')
#organized_dataset.to_csv('organized_dataset.csv', index=False)


dataset = pd.read_csv('18 TorchRL Ads/organized_dataset.csv')
dataset.head()

def get_entry_from_dataset(df, index):
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

    keywords_amount = get_entry_from_dataset.keywords_amount
    return df.iloc[index * keywords_amount:index * keywords_amount + keywords_amount].reset_index(drop=True)

# Example usage
entry = get_entry_from_dataset(dataset, 0)
print(entry)

entry = get_entry_from_dataset(dataset, 1)
print(entry)


# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset, initial_cash=100000.0, device="cpu"):
        super().__init__(device=device)
        self.initial_cash = initial_cash
        self.dataset = dataset
        self.num_features = len(feature_columns)
        self.num_keywords = get_entry_from_dataset(self.dataset, 0).shape[0]
        #self.action_spec = Bounded(low=0, high=1, shape=(self.num_keywords,), dtype=torch.int, domain="discrete")
        #self.action_spec = MultiCategorical(nvec=[2] * self.num_keywords) # 0 = hold, 1 = buy
        #self.action_spec = Categorical
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
        
        self.reset()

    def _reset(self, tensordict=None):
        self.current_step = 0
        self.holdings = torch.zeros(self.num_keywords, dtype=torch.int, device=self.device) # 0 = not holding, 1 = holding keyword
        self.cash = self.initial_cash
        #sample = self.dataset.sample(1)
        #state = torch.tensor(sample[feature_columns].values, dtype=torch.float32).squeeze()
        # Create the initial observation.
        keyword_features = torch.tensor(get_entry_from_dataset(self.dataset, self.current_step)[feature_columns].values, dtype=torch.float32, device=self.device)
        obs = TensorDict({
            "keyword_features": keyword_features,  # Current pki for each keyword
            "cash": torch.tensor(self.cash, dtype=torch.float32, device=self.device),  # Current cash balance
            "holdings": self.holdings.clone()  # 1 for each keyword if we are holding
        }, batch_size=[])
        #return TensorDict({"observation": state}, batch_size=[])
        # step_count initialisieren
        if tensordict is None:
            tensordict = TensorDict({
                "done": torch.tensor(False, dtype=torch.bool, device=self.device),
                "observation": obs,
                "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
                "terminated": torch.tensor(False, dtype=torch.bool, device=self.device),
                "truncated": torch.tensor(False, dtype=torch.bool, device=self.device)
            },
            batch_size=[])
        else:
            tensordict["done"] = torch.tensor(False, dtype=torch.bool, device=self.device)
            tensordict["observation"] = obs
            tensordict["step_count"] = torch.tensor(self.current_step, dtype=torch.int64, device=self.device)
            tensordict["terminated"] = torch.tensor(False, dtype=torch.bool, device=self.device)
            tensordict["truncated"] = torch.tensor(False, dtype=torch.bool, device=self.device)
        
        self.obs = obs
        #print(result)
        print(f'Reset: Step: {self.current_step}')
        return tensordict


    def _step(self, tensordict):
        # Get the action from the input tensor dictionary. 
        action = tensordict["action"]
        #action_idx = action.argmax(dim=-1).item()  # Get the index of the selected keyword
        true_indices = torch.nonzero(action, as_tuple=True)[0]
        action_idx = true_indices[0] if len(true_indices) > 0 else self.action_spec.n - 1

        current_pki = get_entry_from_dataset(self.dataset, self.current_step)
        #action = tensordict["action"].argmax(dim=-1).item()
        
        # Update holdings based on action (only one keyword is selected)
        new_holdings = torch.zeros_like(self.holdings)
        if action_idx < self.num_keywords:
            new_holdings[action_idx] = 1
        self.holdings = new_holdings

        # Calculate the reward based on the action taken.
        reward = self._compute_reward(action, current_pki, action_idx)

         # Move to the next time step.
        self.current_step += 1
        terminated = self.current_step >= (len(self.dataset) // self.num_keywords) - 2 # -2 to avoid going over the last index
        truncated = False

        # Get next pki for the keywords
        next_keyword_features = torch.tensor(get_entry_from_dataset(self.dataset, self.current_step)[feature_columns].values, dtype=torch.float32, device=self.device)
        # todo: most probably we need to remove some columns from the state so we only have the features for the agent to see... change it also in reset
        next_obs = TensorDict({
            "keyword_features": next_keyword_features,  # next pki for each keyword
            "cash": torch.tensor(self.cash, dtype=torch.float32, device=self.device),  # Current cash balance
            "holdings": self.holdings.clone()
        }, batch_size=[])
        
        # Update the state
        self.obs = next_obs
        print(f'Step: {self.current_step}, Action: {action_idx}, Reward: {reward}')
        tensordict["done"] = torch.tensor(terminated or truncated, dtype=torch.bool, device=self.device)
        
        tensordict["done"] = torch.tensor(terminated or truncated, dtype=torch.bool, device=self.device)
        tensordict["observation"] = self.obs
        tensordict["reward"] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        tensordict["step_count"] = torch.tensor(self.current_step-1, dtype=torch.int64, device=self.device)
        tensordict["terminated"] = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        tensordict["truncated"] = torch.tensor(truncated, dtype=torch.bool, device=self.device)
        next = TensorDict({
            "done": torch.tensor(terminated or truncated, dtype=torch.bool, device=self.device),
            "observation": next_obs,
            "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
            "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
            "terminated": torch.tensor(terminated, dtype=torch.bool, device=self.device),
            "truncated": torch.tensor(truncated, dtype=torch.bool, device=self.device)

        }, batch_size=tensordict.batch_size)
        
        return next
    
        

    def _compute_reward(self, action, current_pki, action_idx):
        """Compute reward based on the selected keyword's metrics"""
        if action_idx == self.num_keywords:
            return 0.0
        
        reward = 0.0
        # Iterate thourh all keywords
        for i in range(self.num_keywords):
            sample = current_pki.iloc[i]
            cost = sample["ad_spend"]
            ctr = sample["paid_ctr"]
            if action[i] == True and cost > 5000:
                reward += 1.0
            elif action[i] == False and ctr > 0.15:
                reward += 1.0
            else:
                reward -= 1.0
        return reward

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

# Initialize Environment
env = AdOptimizationEnv(dataset, device=device)
state_dim = env.num_features
#action_dim = env.action_spec.n




# In[ ]:


env.action_spec


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiStockQValueNet(nn.Module):
    def __init__(self, input_dim, num_keywords, num_actions):
        """
        input_dim: Dimension of the input features (e.g., state dimension)
        num_keywords: Number of keywords (each with its own discrete action space)
        num_actions: Number of discrete actions per keyword (e.g., 2 for buy or wait)
        """
        super().__init__()
        # Shared feature extraction backbone.
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Create a separate head for each stock.
        self.heads = nn.ModuleList([nn.Linear(128, num_actions) for _ in range(num_keywords)])
        
    def forward(self, x):
        # x shape: (batch, input_dim)
        features = self.shared(x)  # Shape: (batch, 128)
        # Get Q-values for each stock
        q_values = [head(features) for head in self.heads]  # Each has shape: (batch, num_actions)
        # Stack to form a tensor with shape: (batch, num_stocks, num_actions)
        q_values = torch.stack(q_values, dim=1)
        return q_values

# Example usage:
# Let's assume:
#   - Your environment's state dimension is 20.
#   - You have 3 stocks.
#   - For each stock, there are 3 possible actions.
input_dim = 20
num_stocks = 3
num_actions = 3

q_net = MultiStockQValueNet(input_dim, num_stocks, num_actions)
dummy_input = torch.randn(4, input_dim)  # batch of 4
print(q_net(dummy_input).shape)  # Expected shape: (4, 3, 3)

 # Create a preprocessing layer to flatten and combine inputs
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

flatten_module = TensorDictModule(
    FlattenInputs(),
    in_keys=[("observation", "keyword_features"), ("observation", "cash"), ("observation", "holdings")],
    out_keys=["flattened_input"]
)

from torchrl.modules import EGreedyModule, MLP, QValueModule

# Define dimensions
feature_dim = len(feature_columns)
num_keywords = env.num_keywords
action_dim = env.action_spec.shape[-1]
total_input_dim = feature_dim * num_keywords + 1 + num_keywords  # features + cash + holdings

value_mlp = MLP(in_features=total_input_dim, out_features=action_dim, num_cells=[128, 64])
#value_net = TensorDictModule(value_mlp, in_keys=["observation"], out_keys=["action_value"])
value_net = TensorDictModule(value_mlp, in_keys=["flattened_input"], out_keys=["action_value"])
policy = TensorDictSequential(flatten_module, value_net, QValueModule(spec=env.action_spec))
#policy = TensorDictSequential(value_net, MultiStockQValueNet(len(feature_columns), env.num_keywords, 2))
# Make sure your policy is on the correct device
policy = policy.to(device)

exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
exploration_module = exploration_module.to(device)
policy_explore = TensorDictSequential(policy, exploration_module).to(device)


# In[ ]:


value_mlp


# In[ ]:


from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

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
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

from torch.optim import Adam


# In[ ]:


from torchrl.objectives import DQNLoss, SoftUpdate
#actor = QValueActor(value_net, in_keys=["observation"], action_space=spec)
loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True).to(device)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)


# In[ ]:


import time
total_count = 0
total_episodes = 0
t0 = time.time()
for i, data in enumerate(collector):
    # Write data in replay buffer
    print(f'data: step_count: {data["step_count"]}')
    rb.extend(data.to(device))
    #max_length = rb[:]["next", "step_count"].max()
    max_length = rb[:]["step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            # Make sure sample is on the correct device
            sample = sample.to(device)  # Move the sample to the specified device
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 10 == 0: # Fixed condition (was missing '== 0')
                print(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    #if max_length > 200:  #that is still from the sample where 200 is a good value to balance the CartPole
    #    break
    if total_count > 10_000:
        break

t1 = time.time()

print(
    f"Finished after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)


''''
Todo:
- Clean up the code
- Split training and test data
- Implement tensorbaord
- Implement the visualization
- Implement the saving of the model
- Implement the inference
- Implement the optuna hyperparameter tuning
'''''





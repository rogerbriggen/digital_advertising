#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import optuna
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from copy import deepcopy

# Import functions and classes from digital_advertising.py
from digital_advertising import (
    AdOptimizationEnv, generate_synthetic_data, create_policy,
    split_dataset_by_ratio, learn
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def objective(trial, dataset):
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Sample hyperparameters
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),                          # Learning rate for the optimizer
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),      # Batch size for training
        'exploration_eps_init': trial.suggest_float('exploration_eps_init', 0.5, 1.0),  # Initial value for epsilon in epsilon-greedy exploration
        'exploration_eps_end': trial.suggest_float('exploration_eps_end', 0.01, 0.1),   # Final value for epsilon in epsilon-greedy exploration
        'softupdate_eps': trial.suggest_float('softupdate_eps', 0.9, 0.99),             # Soft update rate for target network
        'gamma': trial.suggest_float('gamma', 0.9, 0.99),                               # Discount factor for future rewards
        'weight_decay':  trial.suggest_float('weight_decay', 1e-6, 1e-4)                # Weight decay for regularization
    }
    
    # Run training with the sampled hyperparameters
    best_reward = learn(params)
    
    return best_reward

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Digital Advertising RL")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials")
    args = parser.parse_args()
    
    # Load dataset
    dataset = generate_synthetic_data(1000)
    
    # Create Optuna study
    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna/digital_ad1.db")
    
    print(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, dataset), n_trials=args.n_trials)
    
    print("Optimization completed!")
    print("Best hyperparameters:")
    for param_name, param_value in study.best_params.items():
        print(f"  {param_name}: {param_value}")

if __name__ == "__main__":
    main()


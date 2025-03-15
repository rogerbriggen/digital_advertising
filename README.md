# Digital Advertising

This project implements a reinforcement learning approach to optimize digital advertising keyword investment decisions. The system learns to make intelligent bidding decisions across multiple keywords, maximizing return on ad spend (ROAS) while operating within budget constraints.

## Project Overview

The project consists of five main components:

1. **Core RL Environment and Training (`digital_advertising.py`)** - Implements the reinforcement learning environment and training pipeline
2. **Hyperparameter Tuning (`hyperparameter_tuning.py`)** - Optimizes model hyperparameters using Optuna
3. **Performance Visualization (`visualize_ad_performance.py`)** - Creates detailed HTML reports with visualizations of model performance
4. **Training Analysis (`tensorboard-analyzer.py`)** - Analyzes TensorBoard logs to provide insights into the training process
5. **Interactive Data Explorer (`analyze_raw_data.py`)** - Provides an interactive Dash web application for exploring and visualizing the raw advertising data

## Link-Collection

- [Analyzing Marketing Performance: Paid Search Campaign](https://medium.com/@farizalfitraaa/analyzing-marketing-performance-paid-search-campaign-6a9ed5f71c7f) ([Dataset](https://www.kaggle.com/datasets/marceaxl82/shopping-mall-paid-search-campaign-dataset))
- [PPC Campaign Performance Data](https://www.kaggle.com/datasets/aashwinkumar/ppc-campaign-performance-data)
- [Google Keywords in Suchkampagnen](https://support.google.com/google-ads/answer/1704371?sjid=14046629692422232983-EU)
- [Google Ads Auktionen](https://support.google.com/google-ads/answer/142918?sjid=14046629692422232983-EU)
- [Goolge Ads KI-gestützte Werbelösung](https://ads.google.com/intl/de_CH/home/campaigns/ai-powered-ad-solutions/)
- [ScaleInsights - Keyword Tracking](https://docs.scaleinsights.com/docs/tracking-keyword-ranking)
- [Google Trends - Keyword Infos](https://trends.google.com/trends/)

## Environment

The conda environment is named torchrl_ads.

## Setup and Installation

On Windows, change the values in environment.yml for the following packages:

- numpy=1.26.4
- pandas=2.2.3

```bash
# Clone the repository
git clone https://github.com/mac-christ/digital_advertising.git
cd digital_advertising

# Create and activate the conda environment
conda env create -f environment.yml
conda activate torchrl_ads

# For CUDA support, run this additional command
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Running the Scripts

### 1. Core RL Training (`digital_advertising.py`)

This script implements the core reinforcement learning environment and training pipeline.

**Key Features:**

- Custom `AdOptimizationEnv` class implementing a TorchRL environment
- Deep Q-Network (DQN) implementation for keyword bidding decisions
- Automatic model saving/loading with best performance tracking
- Synthetic data generation for training and testing

**Usage:**

```bash
python digital_advertising.py
```

**See results in tensorboard**

```bash
# Open a new terminal and activate the environment
cd digital_advertising
conda activate torchrl_ads
# Run tensorboard
tensorboard --logdir=runs
```

Open webbrowser at <http://localhost:6006/> (or check the output of the tensorboard start)

### 2. Hyperparameter Tuning (`hyperparameter_tuning.py`)

Optimizes the model's hyperparameters using Optuna, a hyperparameter optimization framework.

**Key Features:**

- Bayesian optimization for efficient hyperparameter search
- Optimizes learning rate, batch size, discount factor, and exploration parameters
- Reports best hyperparameter configuration for peak performance

**Usage:**

```bash
# Run with default settings (50 trials)
python hyperparameter_tuning.py

# Run with custom number of trials
python hyperparameter_tuning.py --n_trials 100

# Filter output to show only Optuna trial results (Optuna results are moved from St.Err to St.Out)
python hyperparameter_tuning.py 2>&1 | grep -e 'Trial'
```

**See results in tensorboard**

```bash
# Open a new terminal and activate the environment
cd digital_advertising
conda activate torchrl_ads
# Run tensorboard
tensorboard --logdir=runs
```

### 3. Performance Visualization (`visualize_ad_performance.py`)

Creates comprehensive HTML reports with visualizations of the model's performance.

**Key Features:**

- Parses TensorBoard logs to extract training metrics
- Generates insightful visualizations of keyword performance metrics
- Creates decision strategy maps showing investment patterns
- Produces an integrated HTML report with all visualizations

**Usage:**

```bash
# Basic usage with default parameters
python visualize_ad_performance.py

# Specify TensorBoard log directory and output directory
python visualize_ad_performance.py --logdir runs --output_dir visualization_results

# Additional options
python visualize_ad_performance.py --logdir runs --output_dir visualization_results --dataset path/to/dataset.csv --num_samples 2000
```

**Parameters:**

- `--logdir`: Directory containing TensorBoard logs (default: "runs")
- `--output_dir`: Directory to save visualizations (default: "visualization_results")
- `--dataset`: Path to dataset CSV (if None, generates synthetic data)
- `--model`: Optional path to saved model for additional analysis
- `--num_samples`: Number of samples for synthetic data generation (default: 1000)

### 4. Training Analysis (`tensorboard-analyzer.py`)

Provides detailed analysis of the training process from TensorBoard logs.

**Key Features:**

- Extracts and analyzes learning curves and convergence patterns
- Evaluates training stability and performance trends
- Generates specialized visualizations focused on the learning process
- Creates an HTML report summarizing training insights

**Usage:**

```bash
# Basic usage with default parameters
python tensorboard-analyzer.py

# Specify TensorBoard log directory and output directory
python tensorboard-analyzer.py --logdir runs --output_dir training_analysis
```

**Parameters:**

- `--logdir`: Directory containing TensorBoard logs (default: "runs")
- `--output_dir`: Directory to save analysis results (default: "training_analysis")

### 5. Interactive Data Explorer (`analyze_raw_data.py`)

Provides an interactive Dash web application for exploring the raw advertising data and visualizing trends.

**Key Features:**

- Interactive web-based dashboard for data exploration
- Time series visualization of keyword performance metrics
- Percentage change analysis for tracking metric evolution
- Fullscreen visualization options for detailed examination
- Multi-keyword comparison with intuitive color coding

**Usage:**

```bash
python analyze_raw_data.py
```

This will start a local web server at <http://127.0.0.1:8050/> where you can access the interactive dashboard.

## Project Structure

```
digital_advertising/
├── digital_advertising.py        # Core RL environment and training
├── hyperparameter_tuning.py      # Hyperparameter optimization
├── visualize_ad_performance.py   # Performance visualization
├── tensorboard-analyzer.py       # Training process analysis
├── analyze_raw_data.py           # Interactive data exploration dashboard
├── runs                          # Location of saved Tensorboard data
├── saves                         # Location of best model
├── visualization_results         # HTML report
└── environment.yml               # Conda environment specification
```

## Key Technologies

- **PyTorch & TorchRL**: For implementing the reinforcement learning models
- **TensorBoard**: For logging training metrics
- **Optuna**: For hyperparameter optimization
- **Pandas & NumPy**: For data handling and processing
- **Matplotlib & Seaborn**: For data visualization
- **Plotly & Dash**: For interactive data visualization and web dashboards

## Example Workflow

A typical workflow might look like:

1. Run the core training: `python digital_advertising.py`
2. Optimize hyperparameters: `python hyperparameter_tuning.py --n_trials 50`
3. Generate visualizations: `python visualize_ad_performance.py --logdir runs --output_dir visualization_results`
4. Analyze training process: `python tensorboard-analyzer.py --logdir runs --output_dir training_analysis`
5. Interactively explore the raw data: `python analyze_raw_data.py` (visit <http://127.0.0.1:8050/> in your browser)

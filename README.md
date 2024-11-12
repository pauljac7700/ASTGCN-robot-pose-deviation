# ASTGCN for Robotic System Error Prediction

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Building the Graph](#building-the-graph)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Logging and Visualization](#logging-and-visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
This project implements an **Attention-based Spatial-Temporal Graph Convolutional Network (ASTGCN)** for predicting errors in robotic systems. It uses graph neural networks and attention mechanisms to model spatial and temporal dependencies.

## Features
- Spatial and Temporal Attention
- Modular and Configurable Design
- Comprehensive Logging with TensorBoard
- Reproducibility with fixed random seeds

## Architecture
The ASTGCN model consists of:
1. **Graph Construction:** Builds an adjacency matrix for the robotic system.
2. **Data Preparation:** Processes and scales sensor data.
3. **Attention Layers:** Captures correlations between nodes and time steps.
4. **Chebyshev Graph Convolution:** K-order polynomials for graph convolution.
5. **ASTGCN Blocks:** Stacks multiple blocks for spatial-temporal modeling.
6. **Prediction Layer:** Outputs future predictions based on input sequences.

## Installation
### Prerequisites
- Python 3.7+
- CUDA (for GPU support)

### Clone the Repository
```bash
git clone https://github.com/your-username/astgcn-robotic-error-prediction.git
cd astgcn-robotic-error-prediction
```

### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
Adjust settings in `config.yaml` for data paths, model parameters, and training settings.

## Data Preparation
Run `prep_data.py` to process raw data into training, validation, and test sets.

```bash
python prep_data.py
```

## Building the Graph
Construct the adjacency matrix with `build_graph.py`:
```bash
python build_graph.py
```
The graph structure is saved to `data/adjacency_matrix.npy`.

## Training the Model
Train the model with `train.py`:
```bash
python train.py
```
Adjust hyperparameters in `config.yaml`.

## Evaluation
Evaluate the trained model on test data using `test.py`, which computes metrics such as MSE and MAE.

### Running Evaluation
To evaluate the model using `test.py`, ensure that `config.yaml` and your saved model checkpoint are correctly set up, and then run:
```bash
python test.py
```

## Logging and Visualization
Run TensorBoard to view training logs:
```bash
tensorboard --logdir=logs
```

## Usage
To make predictions on new data:
1. Load and preprocess your input sequence.
2. Use the trained model to generate predictions.

## Contributing
1. Fork the Repository
2. Create a Feature Branch
3. Commit Your Changes
4. Push to the Branch
5. Open a Pull Request

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions:
- **Email:** paul.jacobi@rwth-aachen.de
  

  




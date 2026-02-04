# A Simple Graph Neural Network (simple_gnn) Implementation

![output](output.gif)


This repository contains a PyTorch implementation of a Graph Neural Network (GNN) for node classification tasks on graph-structured data. The implementation is based on the PyTorch Geometric library and follows a modular design, separating the code into different modules for better organization and maintainability.

## Project Structure

```
workon_gnn_pytorch/
├── gnn/
│   ├── __init__.py
│   ├── model.py              # GCN model
│   ├── gat_model.py          # GAT model with attention
│   ├── data.py               # Data loading
│   ├── train.py              # Basic training
│   ├── train_with_viz.py     # Training with visualization
│   ├── evaluate.py           # Model evaluation
│   └── visualize.py          # Visualization tools
├── visualizations/           # Generated HTML files
├── main.py                   # Basic training script
├── simple_demo.py            # Visualization demo
├── demo_visualization.py     # Full visualization demo
├── VISUALIZATION_GUIDE.md    # Visualization documentation
├── requirements.txt
├── LICENSE
└── README.md
```

## Overview

Graph Neural Networks (GNNs) are a powerful class of neural networks designed to operate on graph-structured data. They are inspired by Convolutional Neural Networks (CNNs) but operate on graphs instead of grid-like structures like images. The key idea is to learn representations of nodes by iteratively aggregating and updating information from their neighbors.

In this implementation, we use the message passing framework, a common paradigm for GNNs, which involves the following steps for each node:

1. **Aggregate information from neighbors**: Collect the feature vectors of neighboring nodes and edges.
2. **Update node representation**: Use a neural network to update the node's representation based on the aggregated neighborhood information.

This process is repeated for a specified number of iterations, allowing nodes to integrate information from further reaches of the graph.

## Model Architecture

The GNN model implemented in this repository consists of the following layers:

1. **Graph Convolutional Layer (GCNConv)**: This layer performs the message passing operation, aggregating information from neighboring nodes and updating the node representations.
2. **ReLU Activation**: A non-linear activation function (ReLU) is applied to the output of the GCNConv layer.
3. **Linear Layer**: A fully-connected linear layer is used to map the updated node representations to the desired output dimensions (e.g., number of classes for node classification).

## Usage:
Run the following commands on `Windows`:

```
git clone https://github.com/likhith2366/Workon_GNN_Pytorch.git
cd simple_gnn
python -m venv gnn_venv
gnn_venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Run the following commands on `MacOS/Linux`:
```
git clone https://github.com/likhith2366/Workon_GNN_Pytorch.git
cd simple_gnn
python3 -m venv gnn_venv
source gnn_venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

This will load the Cora citation network dataset, train the GNN model, and evaluate its performance on the test set.

## Visualization

Generate interactive 3D visualizations:

```bash
python simple_demo.py
```

This creates two HTML files in `visualizations/`:
- `graph_3d.html` - Interactive 3D graph with nodes and edges
- `training_animation.html` - Animated training progress

Open the HTML files in your browser to interact with the visualizations (drag to rotate, scroll to zoom).

**Models:** GCN (Graph Convolutional Network), GAT (Graph Attention Network)
**Tools:** PyTorch Geometric, NetworkX, Plotly

## Customization

You can customize the GNN model architecture, dataset, and hyperparameters by modifying the respective files:

- `models.py`: Modify the `GNN` class to change the model architecture.
- `data.py`: Update the `load_data` function to load a different dataset or preprocess the data differently.
- `main.py`: Adjust the hyperparameters (e.g., learning rate, hidden dimensions) when instantiating the model, optimizer, and loss function.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


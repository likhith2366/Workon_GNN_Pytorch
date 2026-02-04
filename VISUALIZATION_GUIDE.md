# 🎨 GNN Interactive Visualization Guide

This guide explains how to use the powerful interactive visualization features in this GNN project.

## ✨ Features

### 1. 3D Graph Structure Visualization
- **Interactive 3D graph plots** using Plotly
- Multiple layout algorithms (spring, kamada-kawai, spectral)
- Hover to see node information
- Rotate, zoom, and pan the graph

### 2. Prediction Visualization
- **Color-coded nodes**: Green = correct predictions, Red = incorrect
- Compare predictions vs ground truth
- Filter by training/validation/test sets
- Real-time accuracy display

### 3. Node Embedding Visualization
- **Dimensionality reduction** using t-SNE or PCA
- Visualize high-dimensional embeddings in 3D
- Color by class labels or prediction accuracy
- Track embedding evolution during training

### 4. Animated Training Progress
- **Watch embeddings evolve** as the model trains
- Interactive animation with play/pause controls
- Slider to jump to specific epochs
- Accuracy tracking throughout training

### 5. Attention Weight Visualization (GAT)
- **Visualize attention mechanisms** in Graph Attention Networks
- See which neighbors each node focuses on
- Thickness of edges represents attention weight
- Explore top-k most attended neighbors

## 🚀 Quick Start

### Run the Demo

```bash
python demo_visualization.py
```

This will:
1. Load the Cora dataset
2. Train a GAT model
3. Generate all visualizations
4. Save HTML files you can open in your browser

### Custom Options

```bash
# Use GCN instead of GAT
python demo_visualization.py --model gcn

# Train for more epochs
python demo_visualization.py --epochs 200

# Use a different dataset
python demo_visualization.py --dataset CiteSeer

# Specify output directory
python demo_visualization.py --output-dir my_visualizations
```

## 📊 Using Visualizations in Your Code

### Basic Usage

```python
from gnn.visualize import GraphVisualizer
from gnn.data import load_data

# Load data
data, train_mask, val_mask, test_mask = load_data()

# Create visualizer
visualizer = GraphVisualizer(data, output_dir="my_viz")

# Visualize graph structure
fig = visualizer.visualize_graph_3d(
    node_colors=data.y.numpy(),
    title="My Graph"
)
visualizer.save_figure(fig, "graph.html")
```

### Training with Visualization

```python
from gnn.gat_model import GAT
from gnn.train_with_viz import train_gat_with_visualization
from gnn.visualize import GraphVisualizer

# Initialize model
model = GAT(num_features, 8, num_classes, num_heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Create visualizer
visualizer = GraphVisualizer(data)

# Train with automatic recording
model, history, attention = train_gat_with_visualization(
    model, data, train_mask, val_mask,
    optimizer, criterion,
    visualizer=visualizer,
    num_epochs=200,
    record_every=10  # Record embeddings every 10 epochs
)

# Create animation
fig = visualizer.create_embedding_animation(method='tsne')
visualizer.save_figure(fig, "training_animation.html")
```

### Visualize Predictions

```python
# After training
model.eval()
with torch.no_grad():
    predictions = model(data.x, data.edge_index)

# Visualize all predictions
fig = visualizer.visualize_predictions(predictions)
visualizer.save_figure(fig, "predictions.html")

# Visualize only test set
fig = visualizer.visualize_predictions(
    predictions,
    mask=test_mask,
    title="Test Set Predictions"
)
visualizer.save_figure(fig, "test_predictions.html")
```

### Visualize Embeddings

```python
# Get embeddings from trained model
embeddings = model.get_embeddings(data.x, data.edge_index)

# t-SNE visualization
fig = visualizer.visualize_embeddings(
    embeddings,
    method='tsne',
    predictions=predictions,
    title="Node Embeddings"
)
visualizer.save_figure(fig, "embeddings_tsne.html")

# PCA visualization
fig = visualizer.visualize_embeddings(
    embeddings,
    method='pca',
    predictions=predictions
)
visualizer.save_figure(fig, "embeddings_pca.html")
```

### Attention Weights (GAT only)

```python
from gnn.gat_model import GAT
from gnn.visualize import visualize_attention_weights

# Initialize and train GAT model
model = GAT(num_features, 8, num_classes)
# ... train model ...

# Get attention weights
edge_index, attention = model.get_attention_weights(
    data.x, data.edge_index, layer=1
)

# Visualize attention for specific node
node_idx = 0
temp_data = data.clone()
temp_data.edge_index = edge_index

fig = visualize_attention_weights(
    temp_data,
    attention.squeeze(),
    node_idx,
    k=10  # Show top 10 neighbors
)

# Save
visualizer.save_figure(fig, f"attention_node{node_idx}.html")
```

## 🎯 Advanced Features

### Custom Node Colors

```python
# Color by degree centrality
import networkx as nx
G = nx.from_edgelist(data.edge_index.t().numpy())
centrality = nx.degree_centrality(G)
colors = [centrality[i] for i in range(data.num_nodes)]

fig = visualizer.visualize_graph_3d(
    node_colors=colors,
    title="Graph colored by Degree Centrality"
)
```

### Custom Node Labels

```python
labels = [f"Node {i}\nDegree: {G.degree(i)}"
          for i in range(data.num_nodes)]

fig = visualizer.visualize_graph_3d(
    node_labels=labels,
    title="Graph with Custom Labels"
)
```

### Different Layouts

```python
# Spring layout (force-directed)
fig1 = visualizer.visualize_graph_3d(layout='spring')

# Kamada-Kawai layout
fig2 = visualizer.visualize_graph_3d(layout='kamada_kawai')

# Spectral layout
fig3 = visualizer.visualize_graph_3d(layout='spectral')
```

## 📦 Output Files

All visualizations are saved as **interactive HTML files** that you can:
- Open in any modern web browser
- Share with others (self-contained, no server needed)
- Embed in websites or presentations
- Export as static images (PNG, PDF)

### File Structure

```
visualizations/
├── Cora_graph_structure_3d.html
├── Cora_gat_predictions.html
├── Cora_gat_test_predictions.html
├── Cora_gat_embeddings_tsne.html
├── Cora_gat_embeddings_pca.html
├── Cora_gat_embedding_animation.html
├── Cora_gat_attention_node0.html
├── Cora_gat_attention_node100.html
└── ...
```

## 🎓 Understanding the Visualizations

### Graph Structure
- **Nodes** represent entities (papers, users, molecules, etc.)
- **Edges** connect related nodes
- **Colors** typically represent classes or communities
- **Layout** organizes nodes to reveal structure

### Predictions
- **Green nodes**: Model predicted correctly
- **Red nodes**: Model predicted incorrectly
- Hover to see true label vs prediction
- Test set nodes are often larger

### Embeddings
- Each node becomes a point in 3D space
- Similar nodes cluster together
- Colors show classes or prediction correctness
- t-SNE preserves local structure, PCA preserves global variance

### Attention Weights
- **Thick edges** = high attention (model focuses here)
- **Thin edges** = low attention (less important)
- Red node is the query node
- Blue nodes are neighbors

## 🔧 Troubleshooting

### "No module named 'plotly'"
```bash
pip install plotly networkx scikit-learn kaleido
```

### Visualizations won't open
- Make sure you're opening `.html` files in a web browser
- Try Chrome, Firefox, or Edge (Safari may have issues)

### Animation is slow
- Reduce the number of recorded epochs: `record_every=20`
- Use fewer training epochs
- Use PCA instead of t-SNE for faster computation

### Memory issues with large graphs
- Use smaller datasets (Cora < CiteSeer < PubMed)
- Record less frequently: `record_every=50`
- Visualize a subgraph instead of the full graph

## 🌟 Tips & Best Practices

1. **Start with the demo** to see all features in action
2. **Record every 10-20 epochs** for smooth animations without too much overhead
3. **Use t-SNE for final visualizations**, PCA for quick checks
4. **Visualize attention weights** for interesting nodes (different classes, misclassified nodes)
5. **Compare GCN vs GAT** to see the benefit of attention
6. **Share HTML files** - they work offline and need no setup!

## 📚 Next Steps

- Try different datasets (CiteSeer, PubMed)
- Implement custom GNN architectures
- Add more visualization types (confusion matrices, ROC curves)
- Build a web interface with FastAPI
- Create comparative visualizations between models

---

**Happy Visualizing! 🎉**

For questions or issues, please open an issue on GitHub.

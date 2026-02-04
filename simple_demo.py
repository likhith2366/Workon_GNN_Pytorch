"""
Simple GNN Visualization Demo
Creates just 2 amazing visualizations:
1. 3D Graph Structure (nodes + edges)
2. Animated Training Progress
"""

import torch
from gnn.model import GNN
from gnn.gat_model import GAT
from gnn.data import load_data
from gnn.visualize import GraphVisualizer
from gnn.train_with_viz import train_gat_with_visualization

print("=" * 80)
print("Simple GNN Visualization Demo")
print("=" * 80)

# Load data
print("\n[1/4] Loading Cora dataset...")
data, train_mask, val_mask, test_mask = load_data(dataset_name='Cora')
num_classes = data.y.max().item() + 1
num_features = data.num_node_features

print(f"  - Nodes: {data.num_nodes}")
print(f"  - Edges: {data.num_edges}")
print(f"  - Features: {num_features}")
print(f"  - Classes: {num_classes}")

# Initialize visualizer
print("\n[2/4] Creating visualizer...")
visualizer = GraphVisualizer(data, output_dir="visualizations")

# Visualize the graph structure
print("\n[3/4] Generating 3D Graph Visualization...")
print("  (This shows nodes as points and edges as lines connecting them)")

fig_graph = visualizer.visualize_graph_3d(
    node_colors=data.y.numpy(),
    title="Cora Citation Network - 3D Graph",
    layout='spring'
)
graph_file = visualizer.save_figure(fig_graph, "graph_3d.html")
print(f"  Saved: {graph_file}")
print(f"  -> Open this file in your browser to see the interactive 3D graph!")

# Train model and create animation
print("\n[4/4] Training GAT model and creating animation...")
print("  (This will take about 1 minute)")

model = GAT(num_features, 8, num_classes, num_heads=8, dropout=0.6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Train with visualization recording
model, history, _ = train_gat_with_visualization(
    model, data, train_mask, val_mask, optimizer, criterion,
    visualizer=visualizer,
    num_epochs=50,
    record_every=5,
    verbose=False  # Less output
)

print(f"  Final accuracy: {history['val_acc'][-1]:.2%}")

# Create animation
print("\n  Creating animation of training progress...")
fig_animation = visualizer.create_embedding_animation(method='pca', fps=2)
anim_file = visualizer.save_figure(fig_animation, "training_animation.html")
print(f"  Saved: {anim_file}")
print(f"  -> Watch how node embeddings evolve during training!")

# Final summary
print("\n" + "=" * 80)
print("DONE! Here are your visualizations:")
print("=" * 80)
print(f"\n1. 3D Graph Structure:")
print(f"   Location: {graph_file}")
print(f"   What it shows: Interactive 3D graph with nodes and edges")
print(f"   How to use: Drag to rotate, scroll to zoom, hover over nodes")

print(f"\n2. Training Animation:")
print(f"   Location: {anim_file}")
print(f"   What it shows: How the model learns over time")
print(f"   How to use: Click Play button, use slider to jump to epochs")

print("\n" + "=" * 80)
print("TO VIEW: Double-click any .html file to open in your browser")
print("=" * 80)
print()

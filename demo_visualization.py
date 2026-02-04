"""
Interactive Graph Neural Network Visualization Demo

This script demonstrates all visualization features:
1. 3D Graph Structure Visualization
2. Node Coloring by Predicted vs Actual Labels
3. Real-time Embedding Animation During Training
4. Attention Weight Visualization (GAT)

Usage:
    python demo_visualization.py [--model MODEL] [--epochs EPOCHS] [--dataset DATASET]

Arguments:
    --model: 'gcn' or 'gat' (default: 'gat')
    --epochs: Number of training epochs (default: 100)
    --dataset: 'Cora', 'CiteSeer', or 'PubMed' (default: 'Cora')
"""

import torch
import argparse
import os
from gnn.model import GNN
from gnn.gat_model import GAT
from gnn.data import load_data
from gnn.visualize import GraphVisualizer, visualize_attention_weights
from gnn.train_with_viz import train_model_with_visualization, train_gat_with_visualization


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='GNN Visualization Demo')
    parser.add_argument('--model', type=str, default='gat', choices=['gcn', 'gat'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--dataset', type=str, default='Cora',
                       choices=['Cora', 'CiteSeer', 'PubMed'],
                       help='Dataset to use')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    print("=" * 80)
    print("*** GNN Interactive Visualization Demo ***")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # Load data
    print("\n[*] Loading dataset...")
    data, train_mask, val_mask, test_mask = load_data(dataset_name=args.dataset)
    num_classes = data.y.max().item() + 1
    num_features = data.num_node_features

    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")

    # Initialize visualizer
    print(f"\n[*] Initializing visualizer...")
    visualizer = GraphVisualizer(data, output_dir=args.output_dir)

    # =========================================================================
    # 1. VISUALIZE INITIAL GRAPH STRUCTURE
    # =========================================================================
    print("\n" + "=" * 80)
    print("[1] Generating 3D Graph Structure Visualization")
    print("=" * 80)

    fig_graph = visualizer.visualize_graph_3d(
        node_colors=data.y.numpy(),
        title=f"{args.dataset} Graph Structure (3D)",
        layout='spring'
    )
    graph_path = visualizer.save_figure(fig_graph, f"{args.dataset}_graph_structure_3d.html")
    print(f"[+] Saved to: {graph_path}")

    # =========================================================================
    # 2. TRAIN MODEL WITH VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("[2] Training Model with Real-time Embedding Recording")
    print("=" * 80)

    # Initialize model
    if args.model == 'gcn':
        model = GNN(num_features, 16, num_classes)
        print("Using GCN model")
    else:
        model = GAT(num_features, 8, num_classes, num_heads=8, dropout=0.6)
        print("Using GAT model with attention mechanism")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train with visualization
    print(f"\nTraining for {args.epochs} epochs...")
    if args.model == 'gat':
        trained_model, history, attention_history = train_gat_with_visualization(
            model, data, train_mask, val_mask, optimizer, criterion,
            visualizer=visualizer,
            num_epochs=args.epochs,
            record_every=5,
            verbose=True
        )
    else:
        trained_model, history = train_model_with_visualization(
            model, data, train_mask, val_mask, optimizer, criterion,
            visualizer=visualizer,
            num_epochs=args.epochs,
            record_every=5,
            verbose=True
        )

    # =========================================================================
    # 3. VISUALIZE PREDICTIONS VS GROUND TRUTH
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3] Generating Predictions vs Ground Truth Visualization")
    print("=" * 80)

    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(data.x, data.edge_index)

    # Overall predictions
    fig_predictions = visualizer.visualize_predictions(
        predictions,
        title=f"{args.model.upper()} Predictions vs Ground Truth"
    )
    pred_path = visualizer.save_figure(fig_predictions, f"{args.dataset}_{args.model}_predictions.html")
    print(f"[+] Saved to: {pred_path}")

    # Test set predictions
    fig_test_predictions = visualizer.visualize_predictions(
        predictions,
        mask=test_mask,
        title=f"{args.model.upper()} Test Set Predictions"
    )
    test_pred_path = visualizer.save_figure(fig_test_predictions, f"{args.dataset}_{args.model}_test_predictions.html")
    print(f"✓ Saved to: {test_pred_path}")

    # =========================================================================
    # 4. VISUALIZE NODE EMBEDDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4] Generating Node Embedding Visualizations")
    print("=" * 80)

    if hasattr(trained_model, 'get_embeddings'):
        embeddings = trained_model.get_embeddings(data.x, data.edge_index)
    else:
        embeddings = predictions

    # t-SNE visualization
    print("Generating t-SNE visualization...")
    fig_tsne = visualizer.visualize_embeddings(
        embeddings,
        method='tsne',
        predictions=predictions,
        title=f"{args.model.upper()} Node Embeddings (t-SNE)"
    )
    tsne_path = visualizer.save_figure(fig_tsne, f"{args.dataset}_{args.model}_embeddings_tsne.html")
    print(f"✓ Saved to: {tsne_path}")

    # PCA visualization
    print("Generating PCA visualization...")
    fig_pca = visualizer.visualize_embeddings(
        embeddings,
        method='pca',
        predictions=predictions,
        title=f"{args.model.upper()} Node Embeddings (PCA)"
    )
    pca_path = visualizer.save_figure(fig_pca, f"{args.dataset}_{args.model}_embeddings_pca.html")
    print(f"✓ Saved to: {pca_path}")

    # =========================================================================
    # 5. CREATE EMBEDDING ANIMATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5] Creating Animated Embedding Evolution During Training")
    print("=" * 80)

    print("Generating animation (this may take a moment)...")
    fig_animation = visualizer.create_embedding_animation(method='pca', fps=2)
    anim_path = visualizer.save_figure(fig_animation, f"{args.dataset}_{args.model}_embedding_animation.html")
    print(f"✓ Saved to: {anim_path}")

    # =========================================================================
    # 6. ATTENTION WEIGHT VISUALIZATION (GAT only)
    # =========================================================================
    if args.model == 'gat':
        print("\n" + "=" * 80)
        print("[6] Generating Attention Weight Visualizations")
        print("=" * 80)

        # Get attention weights
        edge_index, attention_weights = trained_model.get_attention_weights(
            data.x, data.edge_index, layer=1
        )

        # Visualize attention for a few interesting nodes
        # Pick nodes from different classes
        unique_classes = data.y.unique()
        sample_nodes = []
        for cls in unique_classes[:3]:  # First 3 classes
            class_nodes = (data.y == cls).nonzero(as_tuple=True)[0]
            if len(class_nodes) > 0:
                sample_nodes.append(class_nodes[0].item())

        for node_idx in sample_nodes:
            print(f"\nVisualizing attention for node {node_idx} (class {data.y[node_idx].item()})...")

            # Create temporary data object with attention weights
            temp_data = data.clone()
            temp_data.edge_index = edge_index

            fig_attention = visualize_attention_weights(
                temp_data,
                attention_weights.squeeze(),
                node_idx,
                k=10
            )

            att_path = visualizer.save_figure(
                fig_attention,
                f"{args.dataset}_gat_attention_node{node_idx}.html"
            )
            print(f"✓ Saved to: {att_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" VISUALIZATION DEMO COMPLETE!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print(f"  1. Graph Structure: {args.dataset}_graph_structure_3d.html")
    print(f"  2. Predictions: {args.dataset}_{args.model}_predictions.html")
    print(f"  3. Test Predictions: {args.dataset}_{args.model}_test_predictions.html")
    print(f"  4. Embeddings (t-SNE): {args.dataset}_{args.model}_embeddings_tsne.html")
    print(f"  5. Embeddings (PCA): {args.dataset}_{args.model}_embeddings_pca.html")
    print(f"  6. Animation: {args.dataset}_{args.model}_embedding_animation.html")
    if args.model == 'gat':
        print(f"  7. Attention weights: {args.dataset}_gat_attention_node*.html")

    print("\n Open any HTML file in your browser to interact with the visualizations!")
    print("=" * 80)

    # Print training summary
    print("\n Training Summary:")
    print(f"  Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {history['val_acc'][-1]:.4f}")

    # Evaluate on test set
    trained_model.eval()
    with torch.no_grad():
        test_out = trained_model(data.x, data.edge_index)
        test_acc = (test_out[test_mask].argmax(dim=1) == data.y[test_mask]).float().mean()
        print(f"  Test Accuracy: {test_acc:.4f}")

    print("\n Enjoy exploring your GNN visualizations!\n")


if __name__ == "__main__":
    main()

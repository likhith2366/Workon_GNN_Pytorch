import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from typing import Optional, List, Tuple


class GraphVisualizer:
    """
    Interactive graph visualization for GNN training and analysis.
    Features:
    - 3D graph structure visualization
    - Node coloring by predicted vs actual labels
    - Real-time embedding animation during training
    - Attention weight visualization
    """

    def __init__(self, data, output_dir: str = "visualizations"):
        """
        Initialize the visualizer with graph data.

        Args:
            data: PyTorch Geometric data object
            output_dir: Directory to save visualizations
        """
        self.data = data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Convert to NetworkX for easier manipulation
        self.G = self._to_networkx()

        # Store training history
        self.embedding_history = []
        self.prediction_history = []
        self.epoch_history = []

    def _to_networkx(self) -> nx.Graph:
        """Convert PyTorch Geometric data to NetworkX graph."""
        G = nx.Graph()

        # Add nodes
        num_nodes = self.data.x.shape[0]
        G.add_nodes_from(range(num_nodes))

        # Add edges
        edge_index = self.data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)

        return G

    def visualize_graph_3d(self,
                          node_colors: Optional[np.ndarray] = None,
                          node_labels: Optional[List[str]] = None,
                          title: str = "3D Graph Structure",
                          layout: str = "spring") -> go.Figure:
        """
        Create interactive 3D visualization of graph structure.

        Args:
            node_colors: Color values for each node
            node_labels: Labels for hover text
            title: Plot title
            layout: NetworkX layout algorithm ('spring', 'kamada_kawai', 'spectral')

        Returns:
            Plotly figure object
        """
        # Compute 3D layout
        if layout == "spring":
            pos = nx.spring_layout(self.G, dim=3, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.G, dim=3)
        elif layout == "spectral":
            pos = nx.spectral_layout(self.G, dim=3)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Extract node positions
        node_xyz = np.array([pos[node] for node in self.G.nodes()])

        # Extract edge positions
        edge_xyz = []
        for edge in self.G.edges():
            x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
            z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]
            edge_xyz.append((x_coords, y_coords, z_coords))

        # Create edge traces
        edge_traces = []
        for edge_x, edge_y, edge_z in edge_xyz:
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=2),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Default colors if not provided
        if node_colors is None:
            node_colors = self.data.y.numpy() if hasattr(self.data, 'y') else np.zeros(len(self.G.nodes()))

        # Default labels if not provided
        if node_labels is None:
            node_labels = [f"Node {i}<br>Label: {self.data.y[i].item()}"
                          for i in range(len(self.G.nodes()))]

        # Create node trace
        node_trace = go.Scatter3d(
            x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Label/Prediction", thickness=20),
                line=dict(color='white', width=0.5)
            ),
            text=node_labels,
            hoverinfo='text',
            name='Nodes'
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=title,
            showlegend=True,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title=''),
            ),
            width=1000,
            height=800,
            hovermode='closest'
        )

        return fig

    def visualize_predictions(self,
                             predictions: torch.Tensor,
                             mask: Optional[torch.Tensor] = None,
                             title: str = "Predictions vs Ground Truth") -> go.Figure:
        """
        Visualize predicted vs actual labels with color coding.

        Args:
            predictions: Model predictions (logits or probabilities)
            mask: Optional mask to highlight specific nodes
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Get predicted classes
        pred_labels = predictions.argmax(dim=1).numpy()
        true_labels = self.data.y.numpy()

        # Compute correctness
        correct = (pred_labels == true_labels).astype(int)

        # Create node labels with prediction info
        node_labels = []
        for i in range(len(pred_labels)):
            status = "✓ Correct" if correct[i] else "✗ Incorrect"
            label_text = (f"Node {i}<br>"
                         f"True Label: {true_labels[i]}<br>"
                         f"Predicted: {pred_labels[i]}<br>"
                         f"Status: {status}")
            if mask is not None and mask[i]:
                label_text += "<br>(Test Set)"
            node_labels.append(label_text)

        # Color nodes: green=correct, red=incorrect
        node_colors = np.where(correct == 1, 'green', 'red')

        # If mask provided, make masked nodes more prominent
        node_sizes = np.ones(len(pred_labels)) * 8
        if mask is not None:
            node_sizes = np.where(mask.numpy(), 12, 6)

        # Create 3D visualization with prediction coloring
        pos = nx.spring_layout(self.G, dim=3, seed=42)
        node_xyz = np.array([pos[node] for node in self.G.nodes()])

        # Edge traces
        edge_traces = []
        for edge in self.G.edges():
            x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
            z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]
            edge_trace = go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines',
                line=dict(color='rgba(125, 125, 125, 0.2)', width=1),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Node trace
        node_trace = go.Scatter3d(
            x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color='white', width=0.5)
            ),
            text=node_labels,
            hoverinfo='text',
            name='Nodes'
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        # Calculate accuracy
        accuracy = correct.mean() * 100

        fig.update_layout(
            title=f"{title}<br>Accuracy: {accuracy:.2f}%",
            showlegend=True,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title=''),
            ),
            width=1000,
            height=800,
            hovermode='closest'
        )

        return fig

    def visualize_embeddings(self,
                            embeddings: torch.Tensor,
                            method: str = 'tsne',
                            predictions: Optional[torch.Tensor] = None,
                            title: str = "Node Embeddings") -> go.Figure:
        """
        Visualize node embeddings in 2D/3D using dimensionality reduction.

        Args:
            embeddings: Node embeddings from model
            method: Dimensionality reduction method ('tsne', 'pca')
            predictions: Optional predictions to show accuracy
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Convert to numpy
        emb_np = embeddings.detach().cpu().numpy()

        # Reduce dimensions
        if method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=3)
        else:
            raise ValueError(f"Unknown method: {method}")

        emb_3d = reducer.fit_transform(emb_np)

        # Determine colors
        if predictions is not None:
            pred_labels = predictions.argmax(dim=1).numpy()
            true_labels = self.data.y.numpy()
            colors = (pred_labels == true_labels).astype(int)
            colorscale = [[0, 'red'], [1, 'green']]
            colorbar_title = "Correct"
        else:
            colors = self.data.y.numpy()
            colorscale = 'Viridis'
            colorbar_title = "Class"

        # Create scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=emb_3d[:, 0],
            y=emb_3d[:, 1],
            z=emb_3d[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title, thickness=20),
                line=dict(color='white', width=0.5)
            ),
            text=[f"Node {i}<br>Class: {self.data.y[i].item()}"
                  for i in range(len(colors))],
            hoverinfo='text'
        )])

        fig.update_layout(
            title=f"{title} ({method.upper()})",
            scene=dict(
                xaxis_title=f'{method.upper()} 1',
                yaxis_title=f'{method.upper()} 2',
                zaxis_title=f'{method.upper()} 3',
            ),
            width=1000,
            height=800
        )

        return fig

    def record_training_step(self,
                            epoch: int,
                            embeddings: torch.Tensor,
                            predictions: torch.Tensor):
        """
        Record embeddings and predictions during training for animation.

        Args:
            epoch: Current epoch number
            embeddings: Node embeddings
            predictions: Model predictions
        """
        self.epoch_history.append(epoch)
        self.embedding_history.append(embeddings.detach().cpu().clone())
        self.prediction_history.append(predictions.detach().cpu().clone())

    def create_embedding_animation(self,
                                  method: str = 'tsne',
                                  fps: int = 5) -> go.Figure:
        """
        Create animated visualization of embeddings evolving during training.

        Args:
            method: Dimensionality reduction method
            fps: Frames per second for animation

        Returns:
            Plotly figure with animation
        """
        if not self.embedding_history:
            raise ValueError("No training history recorded. Use record_training_step() during training.")

        # Reduce all embeddings to 3D
        all_embeddings = torch.cat(self.embedding_history, dim=0).numpy()

        if method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42)
        else:
            reducer = PCA(n_components=3)

        all_reduced = reducer.fit_transform(all_embeddings)

        # Split back into epochs
        num_nodes = self.data.x.shape[0]
        frames_data = []

        for i, epoch in enumerate(self.epoch_history):
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            emb_3d = all_reduced[start_idx:end_idx]

            # Get predictions for this epoch
            pred_labels = self.prediction_history[i].argmax(dim=1).numpy()
            true_labels = self.data.y.numpy()
            correct = (pred_labels == true_labels).astype(int)

            frames_data.append({
                'x': emb_3d[:, 0],
                'y': emb_3d[:, 1],
                'z': emb_3d[:, 2],
                'colors': correct,
                'epoch': epoch,
                'accuracy': correct.mean() * 100
            })

        # Create initial frame
        initial_frame = frames_data[0]

        fig = go.Figure(
            data=[go.Scatter3d(
                x=initial_frame['x'],
                y=initial_frame['y'],
                z=initial_frame['z'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=initial_frame['colors'],
                    colorscale=[[0, 'red'], [1, 'green']],
                    showscale=True,
                    colorbar=dict(title="Correct", thickness=20),
                    line=dict(color='white', width=0.5)
                ),
                text=[f"Node {i}" for i in range(len(initial_frame['x']))],
                hoverinfo='text'
            )],
            layout=go.Layout(
                title=f"Epoch: {initial_frame['epoch']} | Accuracy: {initial_frame['accuracy']:.2f}%",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 1000//fps, "redraw": True},
                                        "fromcurrent": True}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}])
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )],
                sliders=[dict(
                    active=0,
                    yanchor="top",
                    y=0.02,
                    xanchor="left",
                    currentvalue=dict(
                        prefix="Epoch: ",
                        visible=True,
                        xanchor="right"
                    ),
                    steps=[dict(
                        args=[[f.get('epoch')],
                              dict(frame=dict(duration=1000//fps, redraw=True),
                                   mode="immediate")],
                        label=str(f.get('epoch')),
                        method="animate"
                    ) for f in frames_data]
                )],
                scene=dict(
                    xaxis_title=f'{method.upper()} 1',
                    yaxis_title=f'{method.upper()} 2',
                    zaxis_title=f'{method.upper()} 3',
                ),
                width=1000,
                height=800
            ),
            frames=[go.Frame(
                data=[go.Scatter3d(
                    x=f['x'],
                    y=f['y'],
                    z=f['z'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=f['colors'],
                        colorscale=[[0, 'red'], [1, 'green']],
                        showscale=True,
                        line=dict(color='white', width=0.5)
                    )
                )],
                layout=go.Layout(
                    title=f"Epoch: {f['epoch']} | Accuracy: {f['accuracy']:.2f}%"
                ),
                name=str(f['epoch'])
            ) for f in frames_data]
        )

        return fig

    def save_figure(self, fig: go.Figure, filename: str):
        """Save figure as HTML file."""
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved visualization to: {filepath}")
        return filepath


def visualize_attention_weights(data,
                                attention_weights: torch.Tensor,
                                node_idx: int,
                                k: int = 10) -> go.Figure:
    """
    Visualize attention weights for a specific node (requires GAT model).

    Args:
        data: PyTorch Geometric data object
        attention_weights: Attention weights from GAT layer
        node_idx: Index of node to visualize
        k: Number of top neighbors to highlight

    Returns:
        Plotly figure object
    """
    # Get edges and attention for this node
    edge_index = data.edge_index
    node_edges = (edge_index[0] == node_idx).nonzero(as_tuple=True)[0]

    neighbors = edge_index[1, node_edges].numpy()
    attentions = attention_weights[node_edges].detach().cpu().numpy()

    # Get top-k neighbors (or all neighbors if less than k)
    actual_k = min(k, len(attentions))
    top_k_indices = np.argsort(attentions)[-actual_k:]
    top_neighbors = neighbors[top_k_indices]
    top_attentions = attentions[top_k_indices]

    # Create subgraph
    G = nx.Graph()
    G.add_node(node_idx)

    for neighbor, attention in zip(top_neighbors, top_attentions):
        G.add_edge(node_idx, neighbor, weight=attention)

    # Layout
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Create visualization
    edge_traces = []
    for edge in G.edges():
        weight = G[edge[0]][edge[1]]['weight']
        x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
        z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]

        edge_trace = go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines',
            line=dict(
                color=f'rgba(255, 0, 0, {weight})',
                width=weight * 10
            ),
            hoverinfo='text',
            text=f'Attention: {weight:.4f}',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Node trace
    node_xyz = np.array([pos[node] for node in G.nodes()])
    node_colors = ['red' if node == node_idx else 'blue' for node in G.nodes()]
    node_sizes = [15 if node == node_idx else 10 for node in G.nodes()]

    node_trace = go.Scatter3d(
        x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(color='white', width=1)),
        text=[f"Node {node}" for node in G.nodes()],
        textposition="top center",
        hoverinfo='text'
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=f"Attention Weights for Node {node_idx} (Top {k} neighbors)",
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False),
            yaxis=dict(showbackground=False, showticklabels=False),
            zaxis=dict(showbackground=False, showticklabels=False),
        ),
        width=1000,
        height=800,
        showlegend=False
    )

    return fig

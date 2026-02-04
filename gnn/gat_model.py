import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) model.

    GAT learns attention weights between connected nodes, allowing the model
    to focus on the most important neighbors when aggregating information.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_heads: int = 8,
                 dropout: float = 0.6):
        """
        Initialize GAT model.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output classes
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout rate (default: 0.6)
        """
        super(GAT, self).__init__()

        self.dropout = dropout
        self.num_heads = num_heads

        # First GAT layer: multi-head attention
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate attention heads
        )

        # Second GAT layer: output layer
        self.conv2 = GATConv(
            hidden_channels * num_heads,  # Input is concatenated heads
            out_channels,
            heads=1,  # Single head for output
            dropout=dropout,
            concat=False
        )

        # Store attention weights for visualization
        self.attention_weights_layer1 = None
        self.attention_weights_layer2 = None

    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Forward pass through the network.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            return_attention_weights: Whether to return attention weights

        Returns:
            Output logits [num_nodes, out_channels]
            (Optional) Attention weights if return_attention_weights=True
        """
        # First layer with multi-head attention
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention_weights:
            x, (edge_index_1, alpha_1) = self.conv1(
                x, edge_index, return_attention_weights=True
            )
            self.attention_weights_layer1 = alpha_1
        else:
            x = self.conv1(x, edge_index)

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer
        if return_attention_weights:
            x, (edge_index_2, alpha_2) = self.conv2(
                x, edge_index, return_attention_weights=True
            )
            self.attention_weights_layer2 = alpha_2
        else:
            x = self.conv2(x, edge_index)

        if return_attention_weights:
            return F.log_softmax(x, dim=1), {
                'layer1': (edge_index_1, alpha_1),
                'layer2': (edge_index_2, alpha_2)
            }
        else:
            return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index, layer: int = 1):
        """
        Get node embeddings from a specific layer.

        Args:
            x: Node features
            edge_index: Graph connectivity
            layer: Which layer to extract embeddings from (1 or 2)

        Returns:
            Node embeddings
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        if layer == 1:
            return x

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def get_attention_weights(self, x, edge_index, layer: int = 1):
        """
        Get attention weights for visualization.

        Args:
            x: Node features
            edge_index: Graph connectivity
            layer: Which layer's attention to return (1 or 2)

        Returns:
            Tuple of (edge_index, attention_weights)
        """
        self.eval()
        with torch.no_grad():
            _, attention_dict = self.forward(x, edge_index, return_attention_weights=True)

        if layer == 1:
            return attention_dict['layer1']
        else:
            return attention_dict['layer2']


class MultiLayerGAT(torch.nn.Module):
    """
    Multi-layer GAT with flexible architecture.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.6):
        """
        Initialize multi-layer GAT.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features per layer
            out_channels: Number of output classes
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiLayerGAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout))

        # Output layer
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index, layer: int = -2):
        """Get embeddings from specified layer."""
        self.eval()
        with torch.no_grad():
            for i, conv in enumerate(self.convs[:layer+1] if layer > 0 else self.convs[:layer]):
                x = F.dropout(x, p=self.dropout, training=False)
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
        return x

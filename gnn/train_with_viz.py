import torch
from typing import Optional
from gnn.visualize import GraphVisualizer


def train_model_with_visualization(
    model,
    data,
    train_mask,
    val_mask,
    optimizer,
    criterion,
    visualizer: Optional[GraphVisualizer] = None,
    num_epochs: int = 200,
    record_every: int = 10,
    verbose: bool = True
):
    """
    Train model with optional visualization recording.

    Args:
        model: GNN model to train
        data: PyTorch Geometric data object
        train_mask: Training set mask
        val_mask: Validation set mask
        optimizer: Optimizer
        criterion: Loss function
        visualizer: GraphVisualizer instance for recording training
        num_epochs: Number of training epochs
        record_every: Record embeddings every N epochs
        verbose: Print training progress

    Returns:
        Trained model
    """
    model.train()
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'val_acc': [],
        'epochs': []
    }

    for epoch in range(num_epochs):
        # Training step
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_acc = (out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean()

                # Record for visualization
                if visualizer is not None and epoch % record_every == 0:
                    # Get embeddings (if model supports it)
                    if hasattr(model, 'get_embeddings'):
                        embeddings = model.get_embeddings(data.x, data.edge_index)
                    else:
                        # Use penultimate layer output as embeddings
                        embeddings = out

                    visualizer.record_training_step(epoch, embeddings, out)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                # Record history
                training_history['train_loss'].append(loss.item())
                training_history['val_acc'].append(val_acc.item())
                training_history['epochs'].append(epoch)

                if verbose:
                    print(f'Epoch: {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}')

            model.train()

    if verbose:
        print(f'\nBest Validation Accuracy: {best_val_acc:.4f}')

    return model, training_history


def train_gat_with_visualization(
    model,
    data,
    train_mask,
    val_mask,
    optimizer,
    criterion,
    visualizer: Optional[GraphVisualizer] = None,
    num_epochs: int = 200,
    record_every: int = 10,
    verbose: bool = True
):
    """
    Train GAT model with visualization (supports attention weights).

    Args:
        model: GAT model to train
        data: PyTorch Geometric data object
        train_mask: Training set mask
        val_mask: Validation set mask
        optimizer: Optimizer
        criterion: Loss function
        visualizer: GraphVisualizer instance
        num_epochs: Number of training epochs
        record_every: Record embeddings every N epochs
        verbose: Print training progress

    Returns:
        Trained model and attention weights history
    """
    best_val_acc = 0.0
    attention_history = []
    training_history = {
        'train_loss': [],
        'val_acc': [],
        'epochs': []
    }

    for epoch in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Get predictions and attention weights
                out, attention_weights = model(
                    data.x, data.edge_index, return_attention_weights=True
                )

                val_acc = (out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean()

                # Record for visualization
                if visualizer is not None and epoch % record_every == 0:
                    embeddings = model.get_embeddings(data.x, data.edge_index, layer=1)
                    visualizer.record_training_step(epoch, embeddings, out)
                    attention_history.append({
                        'epoch': epoch,
                        'weights': attention_weights
                    })

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                # Record history
                training_history['train_loss'].append(loss.item())
                training_history['val_acc'].append(val_acc.item())
                training_history['epochs'].append(epoch)

                if verbose:
                    print(f'Epoch: {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}')

    if verbose:
        print(f'\nBest Validation Accuracy: {best_val_acc:.4f}')

    return model, training_history, attention_history

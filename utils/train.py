# train.py
#
# The training utilities.
#
# @author: Dung Tran
# @date: September 10, 2025

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import os


def train_step(model, train_dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device, epoch, train_history):
    """
    Perform one training epoch with minibatch training.

    Parameters:
    ----------
    model : nn.Module
        The model to be trained.
    train_dataloader : DataLoader
        The data loader for the training data.
    loss_fn : nn.Module
        The loss function.
    optimizer : optim.Optimizer
        The optimizer.
    device : torch.device
        The device to run the training on.
    loss_history : torch.Tensor
        Tensor to store the loss history.

    Returns:
    -------
    torch.Tensor
        Updated loss history with batch-averaged losses.
    """
    # Train mode
    model.train()
    total_loss = 0
    num_batches = len(train_dataloader)
    
    for batch_idx, data in enumerate(train_dataloader):
        # Zero gradients for each batch
        optimizer.zero_grad()
        
        # Move data to device and unpack
        enc_X, dec_X, labels, enc_valid_lens = [x.to(device) for x in data]
        batch_size = enc_X.size(0)
        
        # Forward pass
        outputs = model(enc_X, dec_X, enc_valid_lens)
        
        # Compute loss (sum reduction to get batch total)
        loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
        batch_avg_loss = loss / batch_size
        
        # Backward pass and optimization
        batch_avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Store batch-averaged loss
        total_loss += batch_avg_loss.item()
        
        # Optional: Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch [{batch_idx + 1}/{num_batches}], Loss: {batch_avg_loss.item():.4f}')

    # Append epoch average loss
    train_history[epoch] = total_loss / num_batches
    print(f'Epoch Average Loss: {train_history[epoch]:.4f}')

    return train_history


def validate_step(model, val_dataloader: DataLoader, loss_fn: nn.Module, device, epoch, val_history):
    """
    Perform one validation epoch with minibatch evaluation.

    Parameters:
    ----------
    model : nn.Module
        The model to be validated.
    val_dataloader : DataLoader
        The data loader for the validation data.
    loss_fn : nn.Module
        The loss function.
    device : torch.device
        The device to run the validation on.
    val_history : torch.Tensor
        List to store the validation loss history.

    Returns:
    -------
    torch.Tensor
        Updated validation history with batch-averaged losses.
    """
    model.eval()
    total_loss = 0
    num_batches = len(val_dataloader)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            # Move data to device and unpack
            enc_X, dec_X, labels, enc_valid_lens = [x.to(device) for x in data]
            batch_size = enc_X.size(0)

            # Forward pass
            outputs = model(enc_X, dec_X, enc_valid_lens)
            
            # Compute loss (sum reduction to get batch total)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
            batch_avg_loss = loss / batch_size
            
            # Store batch-averaged loss
            total_loss += batch_avg_loss.item()
            val_history.append(batch_avg_loss.item())
            
            # Optional: Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Val Batch [{batch_idx + 1}/{num_batches}], Loss: {batch_avg_loss.item():.4f}')

    # Calculate and print epoch average loss
    epoch_avg_loss = total_loss / num_batches
    print(f'Validation Average Loss: {epoch_avg_loss:.4f}')

    val_history[epoch] = epoch_avg_loss
    return val_history


def minibatch_gd(model, train_dataloader: DataLoader, val_dataloader, loss_fn: nn.Module, 
               optimizer: optim.Optimizer, device, num_epochs, log_dir="runs"):
    """
    Perform minibatch gradient descent with TensorBoard visualization.

    Parameters:
    ----------
    model : nn.Module
        The model to be trained.
    train_dataloader : DataLoader
        The data loader for the training data.
    val_dataloader : DataLoader
        The data loader for validation data.
    loss_fn : nn.Module
        The loss function.
    optimizer : optim.Optimizer
        The optimizer.
    device : torch.device
        The device to run the training on.
    num_epochs : int
        Number of epochs to train.
    log_dir : str
        Directory for TensorBoard logs.

    Returns:
    -------
    tuple of (torch.Tensor, torch.Tensor)
        Training and validation loss histories.
    """
    # Initialize histories
    train_history = torch.zeros(num_epochs)
    val_history = torch.zeros(num_epochs)

    # Set up TensorBoard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    writer = SummaryWriter(log_dir)

    # Add graph to TensorBoard
    dummy_input = next(iter(train_dataloader))
    enc_X, dec_X, _, enc_valid_lens = [x.to(device) for x in dummy_input]
    writer.add_graph(model, (enc_X, dec_X, enc_valid_lens))

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        
        # Training step
        train_history = train_step(model, train_dataloader, loss_fn, optimizer, device, epoch, train_history)
        writer.add_scalar('Loss/train', train_history[epoch], epoch)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning rate', current_lr, epoch)
        
        # Validation step
        if val_dataloader is not None:
            val_history = validate_step(model, val_dataloader, loss_fn, device, epoch, val_history)
            writer.add_scalar('Loss/validation', val_history[epoch], epoch)
            
            # Log train/val loss ratio
            loss_ratio = train_history[epoch] / val_history[epoch]
            writer.add_scalar('Loss ratio (train/val)', loss_ratio, epoch)
        
        # Log model parameters distributions and gradients
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # Add custom scalars
        writer.add_scalars('Losses', {
            'train': train_history[epoch],
            'val': val_history[epoch] if val_dataloader else 0
        }, epoch)

    writer.close()
    return train_history, val_history
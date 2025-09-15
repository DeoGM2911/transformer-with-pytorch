# train.py
#
# The training utilities.
#
# @author: Dung Tran
# @date: September 10, 2025

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import os


def train_step(model, dataset, batch_size, loss_fn: nn.Module, optimizer: optim.Optimizer, device, epoch, train_history):
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
    num_batches = dataset.num_train // batch_size
    
    for batch_idx, data in enumerate(dataset.data_loader(batch_size)):
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


def validate_step(model, dataset, batch_size, loss_fn: nn.Module, device, epoch, val_history):
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
    num_batches = dataset.num_val // batch_size
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset.data_loader(batch_size, train=False)):
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
            
            # Optional: Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Val Batch [{batch_idx + 1}/{num_batches}], Loss: {batch_avg_loss.item():.4f}')

    # Calculate and print epoch average loss
    epoch_avg_loss = total_loss / num_batches
    print(f'Validation Average Loss: {epoch_avg_loss:.4f}')

    val_history[epoch] = epoch_avg_loss
    return val_history


def minibatch_gd(model, dataset, batch_size, loss_fn: nn.Module, 
               optimizer: optim.Optimizer, device, num_epochs, writer=None):
    """
    Perform minibatch gradient descent with TensorBoard visualization.

    Parameters:
    ----------
    model : nn.Module
        The model to be trained.
    dataset : Dataset
        The dataset containing training and validation data.
    batch_size : int
        Size of each training batch.
    loss_fn : nn.Module
        The loss function.
    optimizer : optim.Optimizer
        The optimizer.
    device : torch.device
        The device to run the training on.
    num_epochs : int
        Number of epochs to train.
    writer : SummaryWriter, optional
        TensorBoard writer instance for logging.

    Returns:
    -------
    tuple of (torch.Tensor, torch.Tensor)
        Training and validation loss histories.
    """
    # Initialize histories
    train_history = torch.zeros(num_epochs)
    val_history = torch.zeros(num_epochs)

    # Create writer if not provided
    if writer is None:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join('runs', current_time)
        writer = SummaryWriter(log_dir)
        should_close_writer = True
    else:
        should_close_writer = False

    try:
        # Add graph to TensorBoard
        dummy_input = next(iter(dataset.data_loader(batch_size)))
        enc_X, dec_X, _, enc_valid_lens = [x.to(device) for x in dummy_input]
        try:
            writer.add_graph(model, (enc_X, dec_X, enc_valid_lens))
        except Exception as e:
            print(f"Warning: Could not add model graph to TensorBoard: {e}")

        # Training loop
        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            
            # Training step
            train_history = train_step(model, dataset, batch_size, loss_fn, optimizer, device, epoch, train_history)
            writer.add_scalar('Loss/train', train_history[epoch].item(), epoch)
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Validation step
            val_history = validate_step(model, dataset, batch_size, loss_fn, device, epoch, val_history)
            writer.add_scalar('Loss/validation', val_history[epoch].item(), epoch)
            
            # Log train/val loss ratio
            loss_ratio = train_history[epoch].item() / val_history[epoch].item()
            writer.add_scalar('Metrics/loss_ratio', loss_ratio, epoch)
            
            # Log model parameters distributions and gradients (every 5 epochs to save space)
            if epoch % 5 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
            # Ensure data is written to disk
            writer.flush()

    finally:
        # Only close the writer if we created it
        if should_close_writer:
            writer.close()

    return train_history, val_history
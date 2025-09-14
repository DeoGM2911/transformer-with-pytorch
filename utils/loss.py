# loss.py
#
# Contain the loss functions for the transformer model.
#
# @author: Dung Tran
# @date: September 10, 2025

from torch import nn


class MaskedBCELoss(nn.Module):
    """
    Masked binary cross-entropy loss. One can specifies the parameters of nn.BCELoss with **kwargs.
    """
    def __init__(self, pad, **kwargs):
        super(MaskedBCELoss, self).__init__(**kwargs)
        self.pad = pad
        self.loss = nn.CrossEntropyLoss(reduction='none', **kwargs)
    
    def forward(self, pred, label):
        """
        Forward function for the masked binary cross-entropy loss.

        Parameters:
        ----------
        pred : torch.Tensor
            The predictions of the model. Shape: (batch_size, seq_len)
        label : torch.Tensor
            The ground truth labels. Shape: (batch_size, seq_len)

        Returns:
        -------
        torch.Tensor
            The masked binary cross-entropy loss.
        """
        # Create the mask
        mask = (label.reshape(-1) != self.pad).float()
        
        # Compute the loss
        loss = self.loss(pred, label)
        
        # Apply the mask
        loss = loss * mask
        
        # Return the mean loss
        return loss.sum() / mask.sum()

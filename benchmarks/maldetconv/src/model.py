"""
CNN + BiGRU Model Architecture for Malware Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiGRUClassifier(nn.Module):
    """
    CNN + Bidirectional GRU Classifier for malware detection.
    
    Architecture:
    1. Embedding layer (Word2Vec embeddings)
    2. Multiple CNN layers with different kernel sizes
    3. Bidirectional GRU layers
    4. Fully connected classifier
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: list,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        embedding_matrix: torch.Tensor = None
    ):
        super(CNNBiGRUClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        
        # CNN layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=num_filters * len(filter_sizes),  # Concatenated CNN outputs
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        
        # CNN: Conv1d expects (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        # Apply multiple CNN filters
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, seq_len')
            conv_out = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # Global max pooling
            conv_out = conv_out.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(conv_out)
        
        # Concatenate CNN outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Expand for GRU: (batch_size, features) -> (batch_size, 1, features)
        x = x.unsqueeze(1)
        
        # BiGRU
        gru_out, hidden = self.gru(x)  # gru_out: (batch_size, 1, hidden_dim * 2)
        
        # Get last hidden state
        x = gru_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Dropout
        x = self.dropout(x)
        
        # Classifier
        x = self.fc(x)  # (batch_size, num_classes)
        
        return x


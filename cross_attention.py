import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLearnable(nn.Module):
    def __init__(self, embedding_dim):
        """
        Cross-Attention module that learns relationships between source and target sequences
        
        Args:
            embedding_dim: int - embedding dimension (e.g., 128)
        """
        super(CrossAttentionLearnable, self).__init__()
        
        # Linear projections for Keys, Queries and Values
        self.keys_projection = nn.Linear(embedding_dim, embedding_dim)
        self.queries_projection = nn.Linear(embedding_dim, embedding_dim)
        self.values_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Scaling factor for dot product attention
        # Note: The actual scale should be sqrt(embedding_dim), not embedding_dim**2
        self.scale = embedding_dim**0.5
    
    def forward(self, source_sequence, target_sequence):
        """
        Cross-attention mechanism:
        - source_sequence: input sequence (e.g., encoder hidden states from source language)
        - target_sequence: sequence to be updated (e.g., decoder hidden states from target language)
        
        Process:
        1. Project source into Keys (K) and Values (V)
        2. Project target into Queries (Q)
        3. Calculate attention scores between Q and K
        4. Apply softmax to get attention weights
        5. Use weights to create weighted sum of Values (V)
        
        Args:
            source_sequence: tensor of shape [source_len, embedding_dim]
            target_sequence: tensor of shape [target_len, embedding_dim]
            
        Returns:
            updated_target: tensor of shape [target_len, embedding_dim]
        """
        # Create Key projections from source sequence
        # Shape: [source_len, embedding_dim]
        keys = self.keys_projection(source_sequence)
        
        # Create Query projections from target sequence
        # Shape: [target_len, embedding_dim]
        queries = self.queries_projection(target_sequence)
        
        # Create Value projections from source sequence
        # Shape: [source_len, embedding_dim]
        values = self.values_projection(source_sequence)
        
        # Calculate attention scores: Q × K^T / sqrt(d_k)
        # Shape: [target_len, source_len]
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights (probabilities summing to 1 along source dimension)
        # Shape: [target_len, source_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Calculate context vectors: weighted sum of values
        # Shape: [target_len, embedding_dim]
        context_vectors = torch.matmul(attention_weights, values)
        
        return context_vectors

if __name__ == '__main__':
    # Example: Cross-attention between English (source) and French (target) sequences
    # English sequence with 20 tokens, each represented by 128-dimensional embedding
    english_embeddings = torch.randn(20, 128)
    
    # French sequence with 30 tokens, each represented by 128-dimensional embedding
    french_embeddings = torch.randn(30, 128)
    
    # Create cross-attention module
    cross_attention = CrossAttentionLearnable(128)
    
    # Apply cross-attention: update French embeddings with context from English
    updated_french_embeddings = cross_attention(english_embeddings, french_embeddings)
    
    # Print the updated French embeddings
    print(updated_french_embeddings.shape)  # Should be [30, 128]

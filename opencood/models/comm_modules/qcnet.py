"""
QC-Net: Query Control Network for Directed Collaborative Perception

Architecture:
    Input: Communication maps, Pose embeddings, Direction masks
           ↓
    Direction Control Module (3-layer MLP)
           ↓
    Query Confidence Maps (QCMs)
           ↓
    Query Clipping Layer (Top-k selection)
           ↓
    Output: Sparse binary query maps

Usage:
    >>> qcnet = QCNet(input_dim=100*252, hidden_dim=512)
    >>> sparse_queries = qcnet(com_maps, pose_emb, dir_mask, budget=0.2)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionControlModule(nn.Module):
    """
    Direction Control Module: Generates direction-prioritized query confidence maps (QCMs)
    Takes initial query map Q0, pose embeddings, and direction mask as input.
    """
    def __init__(self, input_dim=100*252, hidden_dim=512):
        super(DirectionControlModule, self).__init__()
        
        # Three-layer MLP for direction control
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, Q0, pose_embedding, dir_mask):
        """
        Args:
            Q0: Initial query map [B, C, H, W]
            pose_embedding: Pose information of collaborative CAVs [B, C, H, W]
            dir_mask: Direction mask indicating interested directions [B, C, H, W]
        
        Returns:
            QCMs: Query confidence maps [B, N-1, H, W] for N-1 collaborative CAVs
        """
        B, C, H, W = Q0.shape
        
        # Combine inputs: Apply direction mask and add pose embedding
        # This allows the network to learn which features are important based on direction
        masked_features = Q0 * dir_mask + pose_embedding
        
        # Flatten for MLP processing
        x = masked_features.view(B, -1)  # [B, C*H*W]
        
        # Three-layer MLP with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Reshape back to spatial dimensions
        x = x.view(B, C, H, W)
        
        return x


class QueryClippingLayer(nn.Module):
    """
    Query Clipping Layer: Selects top Q_max × H × W queries based on QCMs
    under communication budget constraints.
    """
    def __init__(self):
        super(QueryClippingLayer, self).__init__()
    
    def forward(self, QCMs, budget):
        """
        Args:
            QCMs: Query confidence maps for each CAV [B, C, H, W]
            budget: Communication budget (0-1 as ratio or absolute number)
        
        Returns:
            sparse_queries: Binary query maps [B, C, H, W] with top-k selected
        """
        B, C, H, W = QCMs.shape
        
        # Calculate maximum number of queries to select
        if budget <= 1.0:  # Budget as ratio
            max_queries = int(budget * H * W)
        else:  # Budget as absolute number
            max_queries = int(budget)
        
        # Flatten spatial dimensions for each sample and channel
        QCMs_flat = QCMs.view(B, C, -1)  # [B, C, H*W]
        
        # For each sample in batch, select top-k queries across all spatial locations
        sparse_queries = torch.zeros_like(QCMs_flat)
        
        for b in range(B):
            # Get all values across channels for this sample
            all_values = QCMs_flat[b].view(-1)  # [C*H*W]
            
            # Select top-k values
            if max_queries >= all_values.numel():
                sparse_queries[b] = 1.0
            else:
                top_k_values, top_k_indices = torch.topk(all_values, max_queries)
                
                # Create flat sparse query map
                sparse_flat = torch.zeros_like(all_values)
                sparse_flat[top_k_indices] = 1.0
                
                # Reshape to [C, H*W]
                sparse_queries[b] = sparse_flat.view(C, -1)
        
        # Reshape back to spatial dimensions
        sparse_queries = sparse_queries.view(B, C, H, W)
        
        return sparse_queries


class QCNet(nn.Module):
    """
    Complete QC-Net implementation with Direction Control Module and Query Clipping Layer.
    
    This network enables ego vehicle to proactively signal its interested directions
    and intelligently select optimal features from collaborative CAVs based on:
    1. Direction priorities (from direction mask)
    2. Communication budget constraints
    3. Pose information of neighboring CAVs
    """
    def __init__(self, input_dim=100*252, hidden_dim=512):
        super(QCNet, self).__init__()
        
        self.direction_control = DirectionControlModule(input_dim, hidden_dim)
        self.query_clipping = QueryClippingLayer()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, com_maps, pairwise_t, dir_mask, budget=0.2):
        """
        Forward pass of QC-Net.
        
        Args:
            com_maps: Initial communication maps (confidence/attention maps) [B, C, H, W]
            pairwise_t: Pairwise transformation matrices (pose embeddings) [B, C, H, W]
            dir_mask: Direction mask from RSU-aided DAS [B, C, H, W]
            budget: Communication budget (default 0.2 means 20% of features)
        
        Returns:
            sparse_query_maps: Binary sparse query maps for selective communication [B, C, H, W]
        """
        B, C, H, W = com_maps.shape
        
        # Ensure all inputs are float tensors and properly shaped
        com_maps = com_maps.float()
        
        # Expand pairwise_t and dir_mask if needed (they might be single sample)
        if pairwise_t.shape[0] != B:
            pairwise_t = pairwise_t.repeat(B, 1, 1, 1).float()
        else:
            pairwise_t = pairwise_t.float()
            
        if dir_mask.shape[0] != B:
            dir_mask = dir_mask.repeat(B, 1, 1, 1).float()
        else:
            dir_mask = dir_mask.float()
        
        # Step 1: Direction Control Module
        # Generate query confidence maps (QCMs) prioritized by direction
        QCMs = self.direction_control(com_maps, pairwise_t, dir_mask)
        
        # Apply sigmoid to get confidence scores in [0, 1]
        QCMs = self.sigmoid(QCMs)
        
        # Step 2: Query Clipping Layer
        # Select top queries based on budget constraints
        sparse_query_maps = self.query_clipping(QCMs, budget)
        
        return sparse_query_maps
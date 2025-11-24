# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
from opencood.models.comm_modules.qcnet import QCNet
from opencood.models.comm_modules.rsu_das import RSUDirectionAttentionScore
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
        self.budget = args['budget']
        self.direction =  args['direction'] if 'direction' in args else None
        self.use_rsu_das = args.get('use_rsu_das', True if self.direction is not None else False)
        
        # For logging
        self.forward_count = 0
        self.log_interval = 100  # Log every 100 forward passes
        
        # Print initialization info
        print("\n" + "="*60)
        print("  Directed-CP Communication Module Initialization")
        print("="*60)
        
        # RSU-aided Direction Attention Score parameters
        if self.use_rsu_das:
            sigma1 = args.get('das_sigma1', 0.1)  # Relative threshold
            sigma2 = args.get('das_sigma2', 0.15)  # Absolute threshold
            self.rsu_das = RSUDirectionAttentionScore(num_directions=4, 
                                                     sigma1=sigma1, 
                                                     sigma2=sigma2)
            print(f"✓ RSU-aided DAS: ENABLED")
            print(f"  - Sigma1 (relative threshold): {sigma1}")
            print(f"  - Sigma2 (absolute threshold): {sigma2}")
        else:
            print(f"✗ RSU-aided DAS: DISABLED")
        
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
            print(f"✓ Gaussian Smoothing: ENABLED (k={kernel_size}, σ={c_sigma})")
        else:
            print(f"✗ Gaussian Smoothing: DISABLED")
        
        # Initialize QC-Net with proper dimensions
        if self.direction is not None:
            self.qcnet = QCNet(input_dim=100*252, hidden_dim=512)
            print(f"✓ QC-Net: ENABLED")
            print(f"  - Input dim: 100x252 = {100*252}")
            print(f"  - Hidden dim: 512")
            print(f"  - Direction priorities: {self.direction}")
        else:
            self.qcnet = None
            print(f"✗ QC-Net: DISABLED (using standard Where2comm)")
        
        print(f"✓ Communication budget: {self.budget} features")
        print(f"✓ Confidence threshold: {self.thre}")
        print("="*60 + "\n")
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)

            if self.direction is None:
                communication_mask = torch.where(communication_maps>self.thre, ones_mask, zeros_mask)
                # Add bandwidth budget to where2comm communication
                count_of_ones = torch.sum(communication_mask == 1).item()
                if count_of_ones > self.budget:
                    indices = (communication_mask == 1).nonzero(as_tuple=True)
                    selected_indices = torch.randperm(indices[0].size(0))[:self.budget]
                    communication_mask[indices[0][selected_indices], 
                        indices[1][selected_indices], 
                        indices[2][selected_indices], 
                        indices[3][selected_indices]] = 0
            else:
                # QC-Net for Directed-CP with RSU-aided DAS
                
                # Step 1: Compute Direction Attention Score (DAS) from RSU or feature map
                if self.use_rsu_das:
                    # Use RSU to compute DAS from traffic density
                    # Extract confidence values to estimate traffic in each direction
                    confidence_numpy = communication_maps[0].detach().cpu().numpy()
                    
                    # If ego interest weights are provided, use them; otherwise use self.direction
                    ego_weights = self.direction if self.direction is not None else None
                    
                    # Compute DAS and direction mask
                    das_scores, das_mask = self.rsu_das.compute_das_from_feature_map(
                        confidence_numpy, 
                        ego_interest_weights=ego_weights
                    )
                    
                    # Log DAS results (only first few times for verification)
                    if self.forward_count < 5:
                        print(f"\n[Directed-CP Forward #{self.forward_count + 1}]")
                        print(f"  RSU-aided DAS Scores (Front-R, Front-L, Back-L, Back-R):")
                        print(f"    {[f'{s:.3f}' for s in das_scores]}")
                        print(f"  Direction Mask: {das_mask}")
                        print(f"  Activated Directions: {sum(das_mask)} / 4")
                    
                    # Create spatial direction mask for BEV feature map
                    spatial_mask = self.rsu_das.create_spatial_direction_mask(
                        communication_maps.shape[2], 
                        communication_maps.shape[3], 
                        das_mask
                    )
                    direction_mask = torch.tensor(spatial_mask, dtype=torch.float32).to(communication_maps.device)
                    direction_mask = direction_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                else:
                    # Fallback: Use manual direction weights if RSU is not enabled
                    direction_weights = self.direction if self.direction is not None else [1, 1, 1, 1]
                    direction_mask = torch.tensor(direction_weights).view(2, 2).to(communication_maps.device)
                    direction_embedding = direction_mask.repeat_interleave(
                        int(communication_maps.shape[2]/2), dim=0
                    ).repeat_interleave(
                        int(communication_maps.shape[3]/2), dim=1
                    )
                    direction_mask = direction_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Step 2: Prepare pose embedding
                pad_size = (0, 249, 0, 50)  # (left, right, top, bottom) 
                pose_embedding = F.pad(
                    pairwise_t_matrix[b, :, :, :, :].view(1, 1, 50, 3), 
                    pad_size, "constant", 0
                ).to(communication_maps.device)  # [1, 1, 100, 252]
                
                # Step 3: Apply QC-Net with Direction Control and Query Clipping
                communication_maps = self.qcnet(
                    communication_maps, 
                    pose_embedding, 
                    direction_mask,
                    budget=self.budget / (H * W)  # Convert absolute budget to ratio
                )
                
                # QC-Net already outputs binary sparse query maps
                communication_mask = torch.where(communication_maps > 0.5, ones_mask, zeros_mask)
                
                # Log QC-Net results
                if self.forward_count < 5:
                    selected_queries = (communication_mask > 0).sum().item()
                    total_queries = communication_mask.numel()
                    sparsity = selected_queries / total_queries
                    print(f"  QC-Net Output:")
                    print(f"    Selected queries: {selected_queries} / {total_queries}")
                    print(f"    Sparsity: {sparsity*100:.2f}%")
                    print(f"    Budget: {self.budget} (ratio: {self.budget/(H*W):.3f})")

            
            communication_rate = communication_mask[0].sum()/(H*W)

            # communication_mask = warp_affine_simple(communication_mask,
            #                                 t_matrix[0, :, :, :],
            #                                 (H, W))
            
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[::2] = ones_mask[::2]

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps*communication_mask_nodiag)

        # save communication map (only for visualization during inference)
        # Uncomment below if you want to save visualization during inference
        # if not self.training:
        #     tensor_data = batch_communication_maps[-1].detach().cpu().numpy()
        #     tensor_data = (tensor_data - tensor_data.min()) / (tensor_data.max() - tensor_data.min() + 1e-8)
        #     fig, axs = plt.subplots(1, 4, figsize=(20, 2))
        #     for i, ax in enumerate(axs):
        #         heatmap = ax.imshow(tensor_data[i].squeeze(), cmap='hot', aspect='auto', vmin=0, vmax=1)
        #         ax.axis('off')
        #     plt.savefig("where2comm_cmp2.png")
        #     plt.close(fig)
        
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        
        # Increment forward counter for logging
        self.forward_count += 1
        
        # Periodic logging
        if self.forward_count % self.log_interval == 0 and self.direction is not None:
            print(f"\n[Directed-CP] Processed {self.forward_count} batches, avg comm rate: {communication_rates:.4f}")
        
        return batch_communication_maps, communication_masks, communication_rates
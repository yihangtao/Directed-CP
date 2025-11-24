"""
RSU-aided Direction Attention Score (DAS) Computation Module

Usage:
    >>> rsu_das = RSUDirectionAttentionScore(num_directions=4, sigma1=0.1, sigma2=0.15)
    >>> das_scores, direction_mask = rsu_das.compute_das_from_feature_map(
    ...     confidence_map, ego_interest_weights=[0.9, 0.9, 0.1, 0.1])
    >>> spatial_mask = rsu_das.create_spatial_direction_mask(H, W, direction_mask)
"""

import torch
import torch.nn as nn
import numpy as np


class RSUDirectionAttentionScore:
    """
    Simulates RSU-aided direction attention score calculation.
    RSU monitors traffic from elevated positions and provides broader views
    than individual CAVs can achieve.
    
    In V2X-Sim 2.0 dataset, we can use agent_0 (if available) or simulate
    RSU by aggregating information from all CAVs.
    """
    
    def __init__(self, num_directions=4, sigma1=1.0, sigma2=1.0):
        """
        Args:
            num_directions: Number of directional sectors (default 4 for quadrants)
            sigma1: Threshold for relative importance (default 1.0)
            sigma2: Threshold for absolute traffic complexity (default 1.0)
        """
        self.num_directions = num_directions
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        # Define direction sectors (angles in degrees)
        # [0°, 90°): Front-right, [90°, 180°): Front-left
        # [180°, 270°): Back-left, [270°, 360°): Back-right
        self.direction_ranges = [
            (0, 90),      # Direction 0: Front-right
            (90, 180),    # Direction 1: Front-left  
            (180, 270),   # Direction 2: Back-left
            (270, 360),   # Direction 3: Back-right
        ]
    
    def compute_das_from_detection(self, detections, ego_pose, ego_interest_weights=None):
        """
        Compute Direction Attention Scores from object detections.
        
        Args:
            detections: List of detected objects with format:
                       [[x, y, z, l, w, h, yaw], ...] in ego coordinate system
                       or dict with 'boxes' key
            ego_pose: Ego vehicle pose [x, y, z, roll, pitch, yaw]
            ego_interest_weights: Optional manual weights from ego CAV [w0, w1, w2, w3]
                                 If None, uses DAS only. If provided, combines with DAS.
        
        Returns:
            das: Direction Attention Scores as list [s0, s1, s2, s3]
            direction_mask: Binary mask for each direction [m0, m1, m2, m3]
        """
        if isinstance(detections, dict):
            if 'boxes' in detections:
                boxes = detections['boxes']
            elif 'vehicles' in detections:
                boxes = detections['vehicles']
            else:
                boxes = []
        else:
            boxes = detections
            
        if len(boxes) == 0:
            # No detections, return uniform distribution
            return [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 0]
        
        # Convert to numpy if tensor
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        boxes = np.array(boxes)
        
        # Count vehicles in each direction
        vehicle_counts = np.zeros(self.num_directions)
        
        for box in boxes:
            x, y = box[0], box[1]
            
            # Calculate angle from ego vehicle to object
            angle = np.arctan2(y, x) * 180.0 / np.pi
            # Normalize to [0, 360)
            angle = (angle + 360) % 360
            
            # Determine which direction sector this belongs to
            for dir_idx, (min_angle, max_angle) in enumerate(self.direction_ranges):
                if min_angle <= angle < max_angle:
                    vehicle_counts[dir_idx] += 1
                    break
        
        # Calculate raw DAS based on vehicle density
        total_vehicles = vehicle_counts.sum()
        if total_vehicles > 0:
            das_raw = vehicle_counts / total_vehicles
        else:
            das_raw = np.ones(self.num_directions) / self.num_directions
        
        # If ego interest weights are provided, combine them with DAS
        if ego_interest_weights is not None:
            ego_interest_weights = np.array(ego_interest_weights)
            # Combine RSU observation with ego interests
            das_combined = das_raw * 0.5 + ego_interest_weights * 0.5
        else:
            das_combined = das_raw
        
        # Apply dual-threshold to create direction mask
        # Equation (1) from the paper
        direction_mask = []
        total_das = das_combined.sum()
        
        for i in range(self.num_directions):
            # Relative importance check
            if total_das > 0:
                relative_importance = das_combined[i] / total_das
            else:
                relative_importance = 0
            
            # Absolute complexity check  
            absolute_importance = das_combined[i]
            
            # Heaviside step function with dual thresholds
            mask_value = 1 if (relative_importance > self.sigma1 or 
                             absolute_importance > self.sigma2) else 0
            direction_mask.append(mask_value)
        
        # Normalize DAS to sum to 1
        das_normalized = das_combined / das_combined.sum() if das_combined.sum() > 0 else das_combined
        
        return das_normalized.tolist(), direction_mask
    
    def compute_das_from_feature_map(self, confidence_map, ego_interest_weights=None):
        """
        Compute DAS from BEV confidence/feature maps.
        This is useful when working with intermediate representations.
        
        Args:
            confidence_map: BEV confidence map [H, W] or [C, H, W]
            ego_interest_weights: Optional manual weights from ego CAV
        
        Returns:
            das: Direction Attention Scores
            direction_mask: Binary mask for each direction
        """
        if isinstance(confidence_map, torch.Tensor):
            confidence_map = confidence_map.cpu().numpy()
        
        if len(confidence_map.shape) == 3:
            # If multi-channel, take max across channels
            confidence_map = confidence_map.max(axis=0)
        
        H, W = confidence_map.shape
        
        # Divide map into 4 quadrants (2x2 grid)
        h_mid = H // 2
        w_mid = W // 2
        
        quadrants = [
            confidence_map[:h_mid, w_mid:],   # Top-right (Front-right)
            confidence_map[:h_mid, :w_mid],   # Top-left (Front-left)
            confidence_map[h_mid:, :w_mid],   # Bottom-left (Back-left)
            confidence_map[h_mid:, w_mid:],   # Bottom-right (Back-right)
        ]
        
        # Calculate density in each quadrant
        densities = np.array([q.sum() for q in quadrants])
        
        # Normalize to get DAS
        total_density = densities.sum()
        if total_density > 0:
            das_raw = densities / total_density
        else:
            das_raw = np.ones(self.num_directions) / self.num_directions
        
        # Combine with ego interests if provided
        if ego_interest_weights is not None:
            ego_interest_weights = np.array(ego_interest_weights)
            das_combined = das_raw * 0.5 + ego_interest_weights * 0.5
        else:
            das_combined = das_raw
        
        # Apply dual-threshold for direction mask
        direction_mask = []
        total_das = das_combined.sum()
        
        for i in range(self.num_directions):
            relative_importance = das_combined[i] / total_das if total_das > 0 else 0
            absolute_importance = das_combined[i]
            
            mask_value = 1 if (relative_importance > self.sigma1 or 
                             absolute_importance > self.sigma2) else 0
            direction_mask.append(mask_value)
        
        # Normalize
        das_normalized = das_combined / das_combined.sum() if das_combined.sum() > 0 else das_combined
        
        return das_normalized.tolist(), direction_mask
    
    def create_spatial_direction_mask(self, H, W, direction_mask):
        """
        Convert direction mask to spatial BEV mask for feature map masking.
        
        Args:
            H, W: Height and width of BEV feature map
            direction_mask: Binary direction mask [m0, m1, m2, m3]
        
        Returns:
            spatial_mask: Spatial mask [H, W] where each quadrant is masked
                         based on direction_mask
        """
        spatial_mask = np.zeros((H, W))
        
        h_mid = H // 2
        w_mid = W // 2
        
        # Apply mask to each quadrant
        if direction_mask[0]:  # Front-right
            spatial_mask[:h_mid, w_mid:] = 1
        if direction_mask[1]:  # Front-left
            spatial_mask[:h_mid, :w_mid] = 1
        if direction_mask[2]:  # Back-left
            spatial_mask[h_mid:, :w_mid] = 1
        if direction_mask[3]:  # Back-right
            spatial_mask[h_mid:, w_mid:] = 1
        
        return spatial_mask


def simulate_rsu_from_cavs(cav_detections_list, ego_cav_idx=0):
    """
    Simulate RSU by aggregating information from all CAVs.
    This is useful when RSU data is not directly available.
    
    Args:
        cav_detections_list: List of detections from each CAV
        ego_cav_idx: Index of ego CAV (default 0)
    
    Returns:
        aggregated_detections: Combined detections in ego coordinate system
    """
    # Simple aggregation: combine all detections
    # In practice, you would transform to common coordinate frame
    all_boxes = []
    
    for cav_idx, detections in enumerate(cav_detections_list):
        if len(detections) > 0:
            all_boxes.extend(detections)
    
    return all_boxes


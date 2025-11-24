# -*- coding: utf-8 -*-
"""
Direction-Weighted Point Pillar Loss for Directed-CP

This loss function implements the Direction-Weighted Detection Loss (DWLoss)

Configuration:
    dir:
        mode: 'N'  # Non-independent mode (coupled weighting)
        weight: [0.9, 0.9, 0.1, 0.1]  # Per-direction importance weights
        dsigma: 1.0  # Weight normalization factor
        th: 0.1  # Direction activation threshold

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from icecream import ic

class DirectionWeightedPointPillarLoss(nn.Module):
    def __init__(self, args):
        super(DirectionWeightedPointPillarLoss, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']

        self.cls = args['cls']
        self.reg = args['reg']
        self.dir = args['dir']
        
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        # Partition size for directional quadrants
        h, w = 50, 126
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        for i in range(2):
            for j in range(2):
                # Calculate sub-matrix indices for directional regions
                start_h, end_h = i * h, (i + 1) * h
                start_w, end_w = j * w, (j + 1) * w

                # Extract directional sub-matrices
                cls_labls = target_dict['pos_equal_one'][:, start_h:end_h, start_w:end_w, :].reshape(batch_size, -1, 1)
                positives = cls_labls > 0
                negatives = target_dict['neg_equal_one'][:, start_h:end_h, start_w:end_w:, :].reshape(batch_size, -1, 1) > 0
                pos_normalizer = positives.sum(1, keepdim=True).float()

                # Rename prediction keys for compatibility
                if f'psm{suffix}' in output_dict:
                    output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
                if f'rm{suffix}' in output_dict:
                    output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']

                # cls loss
                cls_preds = output_dict[f'cls_preds{suffix}'][:,:, start_h:end_h, start_w:end_w] \
                            .permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 1)
                cls_weights = positives * self.pos_cls_weight + negatives * 1.0
                cls_weights /= torch.clamp(pos_normalizer, min=1.0)
                cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
                sub_cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

                # reg loss
                reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
                reg_preds = output_dict[f'reg_preds{suffix}'][:, :, start_h:end_h, start_w:end_w] \
                            .permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 7)
                reg_targets = target_dict['targets'][:, start_h:end_h, start_w:end_w, :] \
                            .reshape(batch_size, -1, 7)
                reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
                reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
                sub_reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

                # Compute direction-weighted loss
                if self.dir['mode'] == 'N':
                    # Normalize direction weights
                    total_weight = sum(self.dir['weight'])
                    normalized_weights = [w / total_weight for w in self.dir['weight']]

                    # Apply threshold: set to 1 if above threshold 'th', else 0
                    thresholded_weights = [1 if w > self.dir['th'] else 0 for w in normalized_weights]

                    # Add 'dsigma' to each weight element
                    added_weights = [w + self.dir['dsigma'] for w in thresholded_weights]

                    # Re-normalize weights
                    total_added_weight = sum(added_weights)
                    direction_weights = [w / total_added_weight for w in added_weights]

                
                total_reg_loss += sub_reg_loss
                total_cls_loss += sub_cls_loss
                sub_total_loss = sub_reg_loss + sub_cls_loss
                weight = direction_weights[i * 2 + j]
                total_loss += sub_total_loss * weight

        self.loss_dict.update({'total_loss': total_loss.item(),
                            'reg_loss': total_reg_loss.item(),
                            'cls_loss': total_cls_loss.item()})

        return total_loss


    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

   

    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f " % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss


def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss
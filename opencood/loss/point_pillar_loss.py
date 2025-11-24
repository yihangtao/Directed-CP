# -*- coding: utf-8 -*-
# Author: Yifan Lu
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from icecream import ic

class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']

        self.cls = args['cls']
        self.reg = args['reg']

        
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

        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1,  1) > 0
        # cared = torch.logical_or(positives, negatives)
        # cls_labls = cls_labls * cared.type_as(cls_labls)
        # num_normalizer = cared.sum(1, keepdim=True)
        pos_normalizer = positives.sum(1, keepdim=True).float()

        # rename variable 
        if f'psm{suffix}' in output_dict:
            output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
        if f'rm{suffix}' in output_dict:
            output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']

        total_loss = 0

        # cls loss
        cls_preds = output_dict[f'cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

        # reg loss
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

        total_loss += reg_loss + cls_loss

        self.loss_dict.update({'total_loss': total_loss.item(),
                               'reg_loss': reg_loss.item(),
                               'cls_loss': cls_loss.item()})

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
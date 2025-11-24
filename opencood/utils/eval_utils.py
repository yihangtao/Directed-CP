# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        result_stat[iou_thresh]['score'] += det_score.tolist()
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt

def caluclate_partial_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames toward partial directions.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3): 3d or (N, 4, 2): 2d. 
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """

    
    
    det_boxes_quadrants, det_score_quadrants = assign_boxes_to_quadrants(det_boxes, det_score)
    gt_boxes_quadrants, _ = assign_boxes_to_quadrants(gt_boxes)
    directions = ["0-90", "90-180", "180-270", "270-360"]

    for k in range(len(directions)):
        fp_pd = []
        tp_pd = []
        gt_pd = gt_boxes_quadrants[k].shape[0]
        if det_boxes_quadrants[k] is not None:
            # convert bounding boxes to numpy array
            det_boxes = common_utils.torch_tensor_to_numpy(det_boxes_quadrants[k])
            det_score = common_utils.torch_tensor_to_numpy(det_score_quadrants[k])
            gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes_quadrants[k])

            # sort the prediction bounding box by score
            score_order_descend = np.argsort(-det_score)
            det_score = det_score[score_order_descend] # from high to low
            det_polygon_list = list(common_utils.convert_format(det_boxes))
            gt_polygon_list = list(common_utils.convert_format(gt_boxes))

            # match prediction and gt bounding box, in confidence descending order
            for i in range(score_order_descend.shape[0]):
                det_polygon = det_polygon_list[score_order_descend[i]]
                ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

                if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                    fp_pd.append(1)
                    tp_pd.append(0)
                    continue

                fp_pd.append(0)
                tp_pd.append(1)

                gt_index = np.argmax(ious)
                gt_polygon_list.pop(gt_index)
            result_stat[iou_thresh]['score_pd'][k] += det_score.tolist()
        result_stat[iou_thresh]['fp_pd'][k] += fp_pd
        result_stat[iou_thresh]['tp_pd'][k] += tp_pd
        result_stat[iou_thresh]['gt_pd'][k] += gt_pd

def assign_boxes_to_quadrants(boxes, score = None):

    # boxes 形状应为 (N, 8, 3)
    N = boxes.shape[0]

    # 计算每个 box 的 x 和 y 坐标的平均值
    x_means = boxes[:, :, 0].mean(dim=1)
    y_means = boxes[:, :, 1].mean(dim=1)
    
    # 初始化四个象限的列表
    quadrant_1 = []  # 第一象限 (x > 0, y > 0)
    score_1 = []
    quadrant_2 = []  # 第二象限 (x < 0, y > 0)
    score_2 = []
    quadrant_3 = []  # 第三象限 (x < 0, y < 0)
    score_3 = []
    quadrant_4 = []  # 第四象限 (x > 0, y < 0)
    score_4 = []
    
    # 遍历每个 box
    for i in range(N):
        # 根据平均值判断所处象限
        if x_means[i] > 0 and y_means[i] > 0:
            quadrant_1.append(boxes[i])
            if score is not None:
                score_1.append(score[i]) 
        elif x_means[i] < 0 and y_means[i] > 0:
            quadrant_2.append(boxes[i])
            if score is not None:
                score_2.append(score[i]) 
        elif x_means[i] < 0 and y_means[i] < 0:
            quadrant_3.append(boxes[i])
            if score is not None:
                score_3.append(score[i]) 
        elif x_means[i] > 0 and y_means[i] < 0:
            quadrant_4.append(boxes[i])
            if score is not None:
                score_4.append(score[i]) 
    
    # 将四个象限的结果转换为张量
    quadrants = [
        torch.stack(quadrant_1) if quadrant_1 else torch.tensor([]),
        torch.stack(quadrant_2) if quadrant_2 else torch.tensor([]),
        torch.stack(quadrant_3) if quadrant_3 else torch.tensor([]),
        torch.stack(quadrant_4) if quadrant_4 else torch.tensor([])
    ]
    
    if score is not None:
        score = [
            torch.stack(score_1) if score_1 else torch.tensor([]),
            torch.stack(score_2) if score_2 else torch.tensor([]),
            torch.stack(score_3) if score_3 else torch.tensor([]),
            torch.stack(score_4) if score_4 else torch.tensor([])
        ]
    
    return quadrants, score




def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec

def calculate_ap_pd(result_stat, iou):
    """
    Calculate the average precision and recall in partial directions, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]
    ap = []

    for i in range(4):
        
        fp = np.array(iou_5['fp_pd'][i])
        tp = np.array(iou_5['tp_pd'][i])
        score = np.array(iou_5['score_pd'][i])
        assert len(fp) == len(tp) and len(tp) == len(score)

        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()

        gt_total = iou_5['gt_pd'][i]

        if gt_total == 0 or fp == [] or tp == []:
            ap.append(-1)
            continue

        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_total

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap_value, _, _ = voc_ap(rec[:], prec[:])
        ap.append(ap_value)

    return ap


def eval_final_results(result_stat, save_path, infer_info=None):
    dump_dict = {}

    # ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    # ap_30_pd = calculate_ap_pd(result_stat, 0.30)
    ap_50_pd = calculate_ap_pd(result_stat, 0.50)
    ap_70_pd = calculate_ap_pd(result_stat, 0.70)

    dump_dict.update({'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      'ap_50_1': ap_50_pd[0],
                      'ap_50_2': ap_50_pd[1],
                      'ap_50_3': ap_50_pd[2],
                      'ap_50_4': ap_50_pd[3],
                      'ap_70_1': ap_70_pd[0],
                      'ap_70_2': ap_70_pd[1],
                      'ap_70_3': ap_70_pd[2],
                      'ap_70_4': ap_70_pd[3],
                      })
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

    print('AP@IoU=0.5 is %.2f, '
          'AP@IoU=0.7 is %.2f, '
          'AP@PD-IoU=0.5 [0-90] is %.2f, '
          'AP@PD-IoU=0.5 [90-180] is %.2f, '
          'AP@PD-IoU=0.5 [180-270] is %.2f, '
          'AP@PD-IoU=0.5 [270-360] is %.2f, '
          'AP@PD-IoU=0.7 [0-90] is %.2f, '
          'AP@PD-IoU=0.7 [90-180] is %.2f, '
          'AP@PD-IoU=0.7 [180-270] is %.2f, '
          'AP@PD-IoU=0.7 [270-360] is %.2f, ' % (ap_50, ap_70, ap_50_pd[0],  ap_50_pd[3],  ap_50_pd[2],  ap_50_pd[1], ap_70_pd[0],  ap_70_pd[3],  ap_70_pd[2],  ap_70_pd[1]))

    return ap_50, ap_70, ap_50_pd[0],  ap_50_pd[3],  ap_50_pd[2],  ap_50_pd[1], ap_70_pd[0],  ap_70_pd[3],  ap_70_pd[2],  ap_70_pd[1]
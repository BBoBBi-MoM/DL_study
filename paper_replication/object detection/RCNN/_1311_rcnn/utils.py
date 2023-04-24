
import torch
import numpy as np
from collections import Counter

def convert_from_midpoint_to_corners(box):
    # box = [x, y, w, h]
    if isinstance(box, dict):
        box = [box['x'], box['y'], box['w'], box['h']]
        
    x1 = box[0] - box[2] / 2
    y1 = box[1] - box[3] / 2
    x2 = box[0] + box[2] / 2
    y2 = box[1] + box[3] / 2
    return x1, y1, x2, y2

def convert_from_corners_to_midpoint(box):
    # box = [x1, y1, x2, y2]
    if isinstance(box, dict):
        box = [box['x1'], box['y1'], box['x2'], box['y2']]
    
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x, y, w, h

def calculate_iou(box1, box2, box_format='corners'):
    # iou = Area of Overlap / Area of Union

    if box_format == 'midpoint':
        # convert boxes to corners format
        box1 = convert_from_midpoint_to_corners(box1)
        box2 = convert_from_midpoint_to_corners(box2)
        
    if isinstance(box1, list):
        box1 = {'x1': box1[0], 'y1': box1[1], 'x2': box1[2], 'y2': box1[3]}
        box2 = {'x1': box2[0], 'y1': box2[1], 'x2': box2[2], 'y2': box2[3]}
        
    x_left = max(box1['x1'], box2['x1'])
    y_top = max(box1['y1'], box2['y1'])
    x_right = min(box1['x2'], box2['x2'])
    y_bottom = min(box1['y2'], box2['y2'])
    
    # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # calculate Overlapping area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def nms(P, iou_threshold = 0.5):
  # P: list of dicts {'bbox':(x1,y1,x2,y2), 'conf':float, 'class':int}
  conf_list = np.array([x['conf'] for x in P])
  conf_order = (-conf_list).argsort() # apply minus to reverse order !!
  isremoved = [False for _ in range(len(P))]
  keep = []

  for idx in range(len(P)):
    to_keep = conf_order[idx]
    if isremoved[to_keep]:
      continue
    
    # append to keep list
    keep.append(P[to_keep])
    isremoved[to_keep] = True
    # remove overlapping bboxes
    for order in range(idx + 1, len(P)):
      bbox_idx = conf_order[order]
      if isremoved[bbox_idx]==False:  # if not removed yet
        # check overlapping
        iou = calculate_iou(P[to_keep]['bbox'], P[bbox_idx]['bbox'])
        if iou > iou_threshold:
          isremoved[bbox_idx] = True
  return keep

# def nms(bboxes, iou_threshold, threshold, box_format="corners"):
#     """
#     Does Non Max Suppression given bboxes
#     Parameters:
#         bboxes (list): list of lists containing all bboxes with each bboxes
#         specified as [class_pred, prob_score, x1, y1, x2, y2]
#         iou_threshold (float): threshold where predicted bboxes is correct
#         threshold (float): threshold to remove predicted bboxes (independent of IoU)
#         box_format (str): "midpoint" or "corners" used to specify bboxes
#     Returns:
#         list: bboxes after performing NMS given a specific IoU threshold
#     """
#
#     assert type(bboxes) == list
#
#     bboxes = [box for box in bboxes if box[1] > threshold]
#     bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
#     bboxes_after_nms = []
#
#     while bboxes:
#         chosen_box = bboxes.pop(0)
#
#         bboxes = [
#             box
#             for box in bboxes
#             if box[0] != chosen_box[0]
#             or calculate_iou(
#                 torch.tensor(chosen_box[2:]),
#                 torch.tensor(box[2:]),
#                 box_format=box_format,
#             )
#             < iou_threshold
#         ]
#
#         bboxes_after_nms.append(chosen_box)
#
#     return bboxes_after_nms

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20):
    # `pred_boxes` is given in [[{'bbox':{'x1', 'x2', 'y1', 'y2'}, 'class'(int), 'conf'}, ...], ...]
    # `true_boxes` is given in [[{'x1', 'x2', 'y1', 'y2', 'class'(int)}, more boxes...], ...]
    
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    
    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = calculate_iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
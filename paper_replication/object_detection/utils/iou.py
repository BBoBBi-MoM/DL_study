import torch

def intersection_on_union(boxes_preds, boxes_labels, box_format="corners"):
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] /2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] /2
        box1_x2 = boxes_preds[..., 2:3] + boxes_preds[..., 2:3] /2
        box1_y2 = boxes_preds[..., 3:4] + boxes_preds[..., 3:4] /2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] /2
        box2_y1 = boxes_labels[..., 0:1] - boxes_labels[..., 3:4] /2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] /2
        box2_y2 = boxes_labels[..., 0:1] + boxes_labels[..., 3:4] /2

    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 0:1]
        box2_x2 = boxes_labels[..., 0:1]
        box2_y2 = boxes_labels[..., 0:1]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)
    
    width = x2 - x1
    height = y2 - y1
    intersection = width.clamp(0) * height.clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
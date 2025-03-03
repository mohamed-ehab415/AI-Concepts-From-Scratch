import torch
from IOU_from_scratsh import *

def NMS(boxes, confidences_score, IOU_threshod, conf_threshold):
    """
    Non-Maximum Suppression algorithm for object detection.

    This function filters out overlapping bounding boxes by keeping only the ones with
    highest confidence scores and suppressing the overlapping boxes with lower scores.

    Args:
        boxes: Tensor of shape [N, 4] containing bounding box coordinates [x1, y1, x2, y2]
        confidences_score: Tensor of shape [N] containing confidence scores for each box
        IOU_threshod: Threshold for Intersection over Union (IoU) to determine box overlap
        conf_threshold: Minimum confidence score threshold to consider a box

    Returns:
        List of indices of the kept boxes after NMS
    """
    # Filter boxes by confidence threshold
    keep_indices = confidences_score > conf_threshold
    filtered_boxes = boxes[keep_indices]
    filtered_scores = confidences_score[keep_indices]

    # Return empty list if no boxes remain after filtering
    if len(filtered_boxes) == 0:
        return []

    # Sort boxes by confidence scores (highest first)
    sorted_indices = torch.argsort(filtered_scores, descending=True)
    kept_boxes = []

    # Process boxes in order of decreasing confidence
    while len(sorted_indices) > 0:
        # Select the box with highest confidence
        best_index = sorted_indices[0]
        kept_boxes.append(best_index.item())

        # Get remaining boxes
        remaining_indices = sorted_indices[1:]

        # Calculate IoU between the selected box and all remaining boxes
        iou_values = torch.tensor([IOU(filtered_boxes[best_index], filtered_boxes[box_idx])
                                  for box_idx in remaining_indices])

        # Keep only boxes with IoU less than threshold (not heavily overlapping)
        sorted_indices = sorted_indices[1:][iou_values < IOU_threshod]

    return kept_boxes

# Example detection boxes [x1, y1, x2, y2]
boxes = torch.tensor([
    [50, 50, 200, 200],  # Box 1
    [55, 55, 205, 205],  # Box 2 (overlaps with Box 1)
    [10, 10, 180, 180],  # Box 3 (distant from others)
    [52, 52, 202, 202]   # Box 4 (overlaps with Box 1)
], dtype=torch.float32)

# Confidence scores for each detection
scores = torch.tensor([0.95, 0.90, 0.80, 0.88])

# Apply NMS
selected_indices = NMS(boxes, scores, IOU_threshod=0.5, conf_threshold=0.5)
selected_boxes = boxes[selected_indices]

print("Selected indices:", selected_indices)
print("Selected boxes:", selected_boxes)

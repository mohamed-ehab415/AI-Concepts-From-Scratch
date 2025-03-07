import torch
from IOU_from_scratsh import IOU

def mean_average_precision(predictions, ground_truth_boxes, iou_threshold, num_classes=20):
    """
    Calculate Mean Average Precision (mAP) for object detection.

    Args:
        predictions (list): List of predicted bounding boxes.
            Each box is [image_id, class_id, confidence_score, x1, y1, x2, y2]
        ground_truth_boxes (list): List of ground truth bounding boxes.
            Each box is [image_id, class_id, confidence_score, x1, y1, x2, y2]
        iou_threshold (float): Threshold for considering a prediction as correct
        num_classes (int): Number of object classes in the dataset

    Returns:
        float: Mean Average Precision value across all classes
    """
    # List to store average precision for each class
    average_precisions = []

    # Calculate AP for each class
    for class_id in range(num_classes):
        # Filter predictions and ground truths for current class
        class_predictions = [box for box in predictions if box[1] == class_id]
        class_ground_truths = [box for box in ground_truth_boxes if box[1] == class_id]

        # Skip if no ground truth boxes for this class
        if len(class_ground_truths) == 0:
            continue

        # Sort predictions by confidence score (highest first)
        class_predictions.sort(key=lambda x: x[2], reverse=True)

        # Initialize true positives and false positives tensors
        true_positives = torch.zeros(len(class_predictions))
        false_positives = torch.zeros(len(class_predictions))
        total_ground_truths = len(class_ground_truths)

        # Evaluate each prediction
        for pred_idx, prediction in enumerate(class_predictions):
            max_iou = 0
            best_gt_idx = -1

            # Find ground truth box with highest IoU
            for gt_idx, ground_truth in enumerate(class_ground_truths):
                current_iou = IOU(torch.tensor(prediction[3:]), torch.tensor(ground_truth[3:]))
                if current_iou > max_iou:
                    max_iou = current_iou
                    best_gt_idx = gt_idx

            # Check if prediction is correct based on IoU threshold
            if max_iou > iou_threshold:
                # Prediction is correct (true positive)
                true_positives[pred_idx] = 1
                # Remove matched ground truth to prevent multiple matches
                class_ground_truths.pop(best_gt_idx)
            else:
                # Prediction is incorrect (false positive)
                false_positives[pred_idx] = 1

        # Calculate cumulative sum for precision-recall curve
        cumulative_true_positives = torch.cumsum(true_positives, dim=0)
        cumulative_false_positives = torch.cumsum(false_positives, dim=0)

        # Calculate precision and recall values
        precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
        recall = cumulative_true_positives / total_ground_truths

        # Calculate average precision using trapezoidal rule for area under precision-recall curve
        average_precisions.append(torch.trapz(precision, recall))

    # Calculate mean average precision across all classes
    mean_average_precision_value = sum(average_precisions) / len(average_precisions)
    return mean_average_precision_value

# Example usage
true_boxes = [
    [1, 0, 1.0, 50, 50, 150, 150],   # Image 1, Class 0
    [1, 0, 1.0, 200, 200, 300, 300], # Image 1, Class 0
]

pred_boxes = [
    [1, 0, 0.9, 55, 55, 145, 145],   # Image 1, Class 0, Confidence 0.9
    [1, 0, 0.8, 60, 60, 140, 140],   # Image 1, Class 0, Confidence 0.8
    [1, 0, 0.7, 205, 205, 295, 295], # Image 1, Class 0, Confidence 0.7
]

# Compute mAP
mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=1)
print(f"Mean Average Precision: {mAP.item():.4f}")  # Mean Average Precision: 0.2917

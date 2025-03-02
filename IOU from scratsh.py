import torch

def IOU(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (torch.Tensor): First bounding box with format [x1, y1, x2, y2]
                            where (x1, y1) is the top-left corner and
                            (x2, y2) is the bottom-right corner.
        box2 (torch.Tensor): Second bounding box with same format as box1.

    Returns:
        torch.Tensor: IoU score between 0 and 1.
    """
    # Calculate coordinates of intersection box
    x1 = max(box1[0], box2[0])  # Get rightmost left edge
    x2 = min(box1[2], box2[2])  # Get leftmost right edge
    y1 = max(box1[1], box2[1])  # Get bottommost top edge
    y2 = min(box1[3], box2[3])  # Get topmost bottom edge

    # Calculate intersection area, ensuring no negative values if boxes don't overlap by clamp
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate area of each individual box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area by adding both box areas and subtracting intersection area
    union_area = (box1_area + box2_area) - inter_area

    # Calculate and return the IoU metric
    IOU_metric = inter_area / union_area
    return IOU_metric


# Example usage
if __name__ == "__main__":
    # Define two example bounding boxes for testing
    box1 = torch.tensor([1.0, 1.0, 3.0, 3.0])  # Box at position (1,1) with width=2, height=2
    box2 = torch.tensor([2.0, 2.0, 4.0, 4.0])  # Box at position (2,2) with width=2, height=2

    # Calculate and print the IoU between the boxes
    print(f"IoU between boxes: {IOU(box1, box2):.4f}")

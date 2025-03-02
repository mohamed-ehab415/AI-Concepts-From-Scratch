# Intersection over Union (IoU) Calculator

## Description

This repository contains a simple PyTorch implementation for calculating the Intersection over Union (IoU) metric between two bounding boxes. IoU is a common evaluation metric in object detection that measures how much two bounding boxes overlap.

## Function Explanation

The `IOU` function works as follows:

1. **Input**: Two bounding boxes in the format `[x1, y1, x2, y2]` where:
   - `(x1, y1)` represents the top-left corner coordinates
   - `(x2, y2)` represents the bottom-right corner coordinates

2. **Intersection Calculation**:
   - Finds the coordinates of the overlapping rectangle
   - Ensures values are valid (non-negative) using `torch.clamp`
   - Calculates the area of intersection

3. **Union Calculation**:
   - Calculates the area of each individual box
   - Computes the union area by adding both areas and subtracting the intersection

4. **IoU Metric**:
   - Returns the ratio of intersection area to union area
   - Result is always between 0 (no overlap) and 1 (perfect overlap)

## Example Usage

```python
import torch

# Create two bounding boxes
box1 = torch.tensor([1.0, 1.0, 3.0, 3.0])  # Format: [x1, y1, x2, y2]
box2 = torch.tensor([2.0, 2.0, 4.0, 4.0])

# Calculate and print IoU
iou_value = IOU(box1, box2)
print(f"IoU: {iou_value}")  # Outputs approximately 0.1429 (1/7)
```

## Applications

This metric is commonly used in:
- Object detection model evaluation
- Non-maximum suppression
- Tracking algorithms

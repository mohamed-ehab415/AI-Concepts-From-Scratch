import torchvision.models as model 
from torchvision.ops import RoIPool
import torch.nn as nn 
import torch 


class FastRCNN(nn.Module):
    """
    Fast R-CNN implementation for object detection.
    
    Architecture:
    1. Backbone (ResNet50): Extracts feature maps from input images
    2. RoI Pooling: Converts variable-sized RoIs into fixed-size feature vectors
    3. Fully Connected Layers: Process pooled features
    4. Two heads: Classification (object class) and Bounding Box Regression (location refinement)
    """
    
    def __init__(self, num_classes):
        """
        Initialize Fast R-CNN model.
        
        Args:
            num_classes (int): Number of classes including background class
        """
        super(FastRCNN, self).__init__()

        # Load pretrained ResNet50 as feature extractor
        resnet = model.resnet50(pretrained=True)

        # Use ResNet layers up to layer3 (before layer4 and avgpool/fc)
        # This gives us feature maps with 1024 channels
        # The output stride is 16 (input downsampled by factor of 16)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])

        # RoI Pooling layer: converts each RoI to fixed 7x7 spatial size
        # spatial_scale=1/16 because backbone downsamples input by 16x
        # Maps RoI coordinates from original image space to feature map space
        self.roipool = RoIPool(output_size=(7, 7), spatial_scale=1.0 / 16.0)
        
        # After RoI pooling: 1024 channels * 7 * 7 = 50,176 features per RoI

        # Fully connected layers to learn high-level representations
        self.fc1 = nn.Linear(7 * 7 * 1024, 8192)  # First FC layer
        self.fc2 = nn.Linear(8192, 4096)           # Second FC layer

        # Classification head: predicts class probabilities for each RoI
        self.classifier = nn.Linear(4096, num_classes)
        
        # Bounding box regression head: predicts 4 values (dx, dy, dw, dh) per class
        # Output size = num_classes * 4 (class-specific bbox refinements)
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)
    
    def forward(self, images, rois):
        """
        Forward pass through Fast R-CNN.
        
        Args:
            images (Tensor): Input images of shape (batch_size, 3, H, W)
            rois (Tensor): Region proposals of shape (num_rois, 5)
                          Format: [batch_index, x1, y1, x2, y2]
                          batch_index indicates which image in batch this RoI belongs to
        
        Returns:
            class_scores (Tensor): Class predictions of shape (num_rois, num_classes)
            bbox_deltas (Tensor): Bbox refinements of shape (num_rois, num_classes * 4)
        """
        # Step 1: Extract feature maps from images
        # Shape: (batch_size, 1024, H/16, W/16)
        feature_map = self.backbone(images)

        # Step 2: Apply RoI pooling to extract fixed-size features for each RoI
        # Shape: (num_rois, 1024, 7, 7)
        rois_pool = self.roipool(feature_map, rois)
        
        # Step 3: Flatten pooled features for fully connected layers
        # Shape: (num_rois, 1024 * 7 * 7) = (num_rois, 50176)
        rois_pool = rois_pool.reshape(rois_pool.size(0), -1)

        # Step 4: Pass through fully connected layers
        fc1 = self.fc1(rois_pool)  # Shape: (num_rois, 8192)
        fc2 = self.fc2(fc1)        # Shape: (num_rois, 4096)

        # Step 5: Classification head - predict class scores
        # Shape: (num_rois, num_classes)
        class_scores = self.classifier(fc2)
        
        # Step 6: Bounding box regression head - predict bbox adjustments
        # Shape: (num_rois, num_classes * 4)
        # For each class, predicts 4 values: (Δx, Δy, Δw, Δh)
        bbox_deltas = self.bbox_regressor(fc2)
        
        return class_scores, bbox_deltas


if __name__ == "__main__":
    # ========== MODEL INITIALIZATION ==========
    num_classes = 10  # Number of classes (e.g., 9 object classes + 1 background)
    model = FastRCNN(num_classes)

    # ========== PREPARE DUMMY INPUT DATA ==========
    # Simulated input image: 1 image, 3 color channels (RGB), 512x512 pixels
    images = torch.randn(1, 3, 512, 512)
    
    # Region of Interest (RoI) proposals
    # Format: [batch_index, x1, y1, x2, y2]
    # - batch_index: which image in the batch (0 for first image)
    # - (x1, y1): top-left corner coordinates
    # - (x2, y2): bottom-right corner coordinates
    # Here we have 4 RoIs from the same image (all have batch_index=0)
    rois = torch.tensor([[0, 100, 100, 200, 200],  # RoI 1: 100x100 box at (100,100)
                         [0, 50, 100, 200, 210],   # RoI 2: 150x110 box at (50,100)
                         [0, 10, 35, 78, 100],     # RoI 3: 68x65 box at (10,35)
                         [0, 10, 20, 70, 70],      # RoI 4: 60x50 box at (10,20)
                         ], dtype=torch.float)

    # ========== FORWARD PASS ==========
    # Run the model: extract features, pool RoIs, and predict classes + bbox refinements
    class_scores, bbox_deltas = model(images, rois)

    # ========== OUTPUT ==========
    # class_scores: Shape (4, 10) - 4 RoIs, 10 class scores each
    # bbox_deltas: Shape (4, 40) - 4 RoIs, 40 bbox adjustments (4 per class × 10 classes)
    print("Class Scores:", class_scores)
    print("Class Scores Shape:", class_scores.shape)  # [4, 10]
    print("\nBounding Box Deltas:", bbox_deltas)
    print("Bounding Box Deltas Shape:", bbox_deltas.shape)  # [4, 40]
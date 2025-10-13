import torch.nn as nn 
import torchvision.models as models 
import torch
import torch.optim as optim

# =========================
# Multi-Class Object Detector
# =========================
class MultiClassDetector(nn.Module):
    """
    A neural network that predicts both class labels and bounding box coordinates.
    Uses a pretrained ResNet50 backbone.
    """
    def __init__(self, num_classes):
        super(MultiClassDetector, self).__init__()
        
        # Load pretrained ResNet50 model
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer to predict:
        # num_classes for classification + 4 for bounding box coordinates
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes + 4)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            class_scores: predicted class logits (batch_size, num_classes)
            bbox_coords: predicted bounding box coordinates (batch_size, 4)
        """
        output = self.backbone(x)

        # Last 4 outputs are bounding box coordinates
        bbox_coords = output[:, -4:] 

        # Remaining outputs are class scores
        class_scores = output[:, :-4]

        return class_scores, bbox_coords


# =========================
# Combined Loss for Multi-Class Detection
# =========================
class MultiClassCombinedLoss(nn.Module):
    """
    Combines CrossEntropyLoss for classification and MSELoss for bounding boxes
    """
    def __init__(self):
        super(MultiClassCombinedLoss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()  # classification loss
        self.bbox_loss = nn.MSELoss()  # regression loss for bounding boxes

    def forward(self, class_scores, bbox_coords, target_classes, target_bboxes):
        """
        Compute combined loss
        Args:
            class_scores: predicted class logits
            bbox_coords: predicted bounding boxes
            target_classes: ground truth class labels
            target_bboxes: ground truth bounding boxes
        Returns:
            Total loss (classification + regression)
        """
        cls_loss = self.class_loss(class_scores, target_classes)
        reg_loss = self.bbox_loss(bbox_coords, target_bboxes)
        return cls_loss + reg_loss
    

if __name__ == '__main__':
    num_classes = 5  # Example: 5 classes
    model = MultiClassDetector(num_classes=num_classes)
    loss_fn = MultiClassCombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    bs = 10
    inputs = torch.randn(bs, 3, 224, 224)
    target_classes = torch.randint(0, num_classes, (bs,))  # GT
    target_bboxes = torch.rand(bs, 4)  # GT

    num_epochs = 5

    for epoch in range (num_epochs):
        model.train()
        optimizer.zero_grad()
        class_scores, bbox_coords = model(inputs)
        loss = loss_fn(class_scores, bbox_coords, target_classes, target_bboxes)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")




        

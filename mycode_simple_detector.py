import torch.nn as nn 
import torchvision.models as models 
import torch
import torch.optim as optim
class MultiClassDetector(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassDetector, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes + 4)

    def forward(self,x):
         output=self.backbone(x)

         bbox_coords = output[:, -4:] 
         class_scores = output[:, :-4]

         return class_scores,bbox_coords


class MultiClassCombinedLoss(nn.Module):
    def __init__(self):
        super(MultiClassCombinedLoss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()  # for multi-class classification
        self.bbox_loss = nn.MSELoss()  # for bounding box regression
    def forward(self, class_scores, bbox_coords, target_classes, target_bboxes):
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



        
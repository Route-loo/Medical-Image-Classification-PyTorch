import torch
import torch.nn as nn
from torchvision import models


class MedicalImageClassifier(nn.Module):
    """ResNet18 transfer-learning binary classifier."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = MedicalImageClassifier(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("output shape:", out.shape)  # expected: torch.Size([2, 2])

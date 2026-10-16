"""
轻量化模型：MobileNetV2 替换 ResNet18 作为骨干。
参数量约 3.5M（ResNet18 约 11M），适合边缘端部署。
训练时在 train.py 里把导入换成：
    from model_mobile import MobileNetClassifier as MedicalImageClassifier
"""
import torch
import torch.nn as nn
from torchvision import models


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    m = MobileNetClassifier(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("输出形状:", out.shape)  # 预期 torch.Size([2, 2])
    n = sum(p.numel() for p in m.parameters())
    print(f"MobileNetV2 参数量: {n/1e6:.2f}M")

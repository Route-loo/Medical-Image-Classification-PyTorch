import torch
import torch.nn as nn
from torchvision import models

class MedicalImageClassifier(nn.Module):
    """基于ResNet18迁移学习的医学影像二分类模型"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # 加载预训练ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        # 替换最后一层全连接层，适配二分类任务
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # 测试模型输入输出
    model = MedicalImageClassifier(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"模型输出形状: {out.shape}")  # 预期: torch.Size([2, 2])

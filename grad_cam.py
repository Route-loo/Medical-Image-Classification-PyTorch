"""
Grad-CAM 可解释性可视化：在胸片上叠加热力图，标出模型判断时关注的病灶区域。
用法：把要测试的胸片路径传入 grad_cam_on_image()，运行后保存 heatmap.jpg
依赖：pip install opencv-python matplotlib
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataset import get_transforms
from model import MedicalImageClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model.pth"
CLASS_NAMES = ["正常", "肺炎"]

# 1. 加载自己训练好的权重（pretrained=False，不联网下载）
model = MedicalImageClassifier(num_classes=2, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# 2. 选定目标层：ResNet18 最后一个残差块 layer4[-1]
target_layer = model.backbone.layer4[-1]

# 3. 钩子：保存前向特征图与反向梯度
activations, gradients = {}, {}

def forward_hook(module, inp, out):
    activations["value"] = out.detach()

def backward_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


def grad_cam_on_image(image_path, save_path="heatmap.jpg"):
    # 读取原图用于叠加显示
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    raw = cv2.resize(raw, (224, 224))

    # 模型输入预处理
    pil = Image.open(image_path).convert("L")
    tensor = get_transforms("val")(pil).unsqueeze(0).to(DEVICE)

    # 前向 + 对预测类别反向传播
    output = model(tensor)
    pred = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred].backward()

    # 通道权重 = 梯度全局平均池化，再加权求和特征图并 ReLU
    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 热力图叠加
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * heatmap + 0.5 * cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM 关注区域（预测: {CLASS_NAMES[pred]}）")
    plt.axis("off")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"热力图已保存: {save_path}，预测类别: {CLASS_NAMES[pred]}")


if __name__ == "__main__":
    # 替换成你的测试胸片路径
    grad_cam_on_image("./test.jpg")

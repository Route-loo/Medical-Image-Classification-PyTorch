"""
批量 Grad-CAM：自动遍历 chest_xray/test 下所有胸片，
逐张生成热力图并保存到 heatmap/ 文件夹（按真实类别分 NORMAL / PNEUMONIA 子目录）。
用法：pip install opencv-python matplotlib pillow && python grad_cam_batch.py
"""
import os
import torch
import cv2
import numpy as np
from PIL import Image
from dataset import get_transforms
from model import MedicalImageClassifier

# ==================== 配置 ====================
TEST_DIR = "./chest_xray/test"          # 测试集目录（含 NORMAL / PNEUMONIA）
OUT_DIR = "./heatmap"                   # 热力图输出目录
MODEL_PATH = "./checkpoints/best_model.pth"
CLASS_NAMES = ["正常", "肺炎"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# ==============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载自己训练好的权重（pretrained=False，不联网下载）
model = MedicalImageClassifier(num_classes=2, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# 2. 目标层：ResNet18 最后一个残差块
target_layer = model.backbone.layer4[-1]
activations, gradients = {}, {}

def forward_hook(module, inp, out):
    activations["value"] = out.detach()

def backward_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

transform = get_transforms("val")


def grad_cam_one(pil_img):
    """对单张 PIL 图像计算 Grad-CAM，返回叠加热力图（RGB 数组）与预测结果"""
    tensor = transform(pil_img.convert("L")).unsqueeze(0).to(DEVICE)

    output = model(tensor)
    prob = torch.softmax(output, dim=1)
    pred = output.argmax(dim=1).item()
    confidence = prob[0, pred].item()

    model.zero_grad()
    output[0, pred].backward()

    # 通道权重 = 梯度全局平均池化
    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 原图灰度缩放到 224
    raw = np.array(pil_img.convert("L").resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * heatmap + 0.5 * cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)).astype(np.uint8)
    return overlay, pred, confidence


def main():
    # 创建输出目录（按真实类别建子文件夹）
    for cls in os.listdir(TEST_DIR):
        cls_dir = os.path.join(TEST_DIR, cls)
        if os.path.isdir(cls_dir):
            os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

    total = 0
    correct = 0

    for cls in sorted(os.listdir(TEST_DIR)):
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        out_cls_dir = os.path.join(OUT_DIR, cls)

        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            img_path = os.path.join(cls_dir, fname)
            try:
                pil_img = Image.open(img_path)
                overlay, pred, confidence = grad_cam_one(pil_img)
            except Exception as e:
                print(f"⚠️ 跳过 {fname}: {e}")
                continue

            total += 1
            true_label = 0 if cls.upper() == "NORMAL" else 1
            if pred == true_label:
                correct += 1

            stem = os.path.splitext(fname)[0]
            save_name = f"{stem}_pred{CLASS_NAMES[pred]}_conf{confidence:.3f}.jpg"
            import matplotlib.pyplot as plt
            plt.imsave(os.path.join(out_cls_dir, save_name), overlay)

            print(f"[{cls}] {fname} -> 预测:{CLASS_NAMES[pred]} 置信度:{confidence:.3f}")

    print("\n" + "=" * 40)
    if total:
        print(f"共处理 {total} 张，预测正确 {correct} 张，"
              f"批量Grad-CAM准确率: {100*correct/total:.2f}%")
    else:
        print("没有找到图片，请检查 TEST_DIR 路径")
    print(f"热力图已保存到: {os.path.abspath(OUT_DIR)}")
    print("=" * 40)


if __name__ == "__main__":
    main()

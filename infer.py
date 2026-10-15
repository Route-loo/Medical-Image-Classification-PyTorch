import torch
from PIL import Image
from dataset import get_transforms
from model import MedicalImageClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model.pth"
CLASS_NAMES = ["正常", "肺炎"]  # 对应数据集NORMAL / PNEUMONIA

def predict_image(image_path):
    # 加载图像并预处理
    image = Image.open(image_path).convert("L")  # 读取灰度图
    transform = get_transforms("val")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 加载模型
    model = MedicalImageClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)

    pred_class = CLASS_NAMES[pred.item()]
    confidence = prob[0][pred.item()].item()
    print(f"🩺 预测结果: {pred_class}，置信度: {confidence:.4f}")
    return pred_class, confidence

if __name__ == "__main__":
    # 改成你自己的测试图片路径
    predict_image("./test.jpg")

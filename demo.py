import torch
import gradio as gr
from PIL import Image
from dataset import get_transforms
from model import MedicalImageClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model.pth"
CLASS_NAMES = ["正常", "肺炎"]

# 加载模型
model = MedicalImageClassifier(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
transform = get_transforms("val")


def predict(image):
    image = Image.fromarray(image).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)

    pred_label = CLASS_NAMES[pred.item()]
    confidence = prob[0][pred.item()].item()
    return {
        "预测结果": pred_label,
        "置信度": f"{confidence:.4f}",
        "说明": "结果仅供参考，不构成医疗诊断"
    }


# 构建界面
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="上传胸部X光片"),
    outputs=gr.JSON(label="预测结果"),
    title="胸部X光肺炎检测系统",
    description="基于ResNet18迁移学习的医学影像二分类模型 | AUC=0.9723",
    examples=["./test.jpg"]  # 可以放一张示例图
)

iface.launch(share=True)
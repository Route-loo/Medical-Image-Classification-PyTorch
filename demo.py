import torch
import gradio as gr
from PIL import Image
from dataset import get_transforms
from model import MedicalImageClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model.pth"
CLASS_NAMES = ["正常", "肺炎"]

# 加载自己训练好的权重，pretrained=False 不联网下载
model = MedicalImageClassifier(num_classes=2, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
transform = get_transforms("val")


def predict(image):
    image = Image.fromarray(image).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)

    pred_idx = pred.item()
    return {
        "预测结果": CLASS_NAMES[pred_idx],
        "置信度": f"{prob[0][pred_idx].item():.4f}",
        "说明": "结果仅供参考，不构成医疗诊断",
    }


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="上传胸部X光片"),
    outputs=gr.JSON(label="预测结果"),
    title="胸部X光肺炎检测系统",
    description="基于ResNet18迁移学习的医学影像二分类模型",
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860, share=False)

import torch
from PIL import Image
from dataset import get_transforms
from model import MedicalImageClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model.pth"
CLASS_NAMES = ["Normal", "Pneumonia"]  # matches NORMAL / PNEUMONIA folders


def predict_image(image_path):
    # Load and preprocess: grayscale
    image = Image.open(image_path).convert("L")
    transform = get_transforms("val")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Load trained weights (pretrained=False: no online download)
    model = MedicalImageClassifier(num_classes=2, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)

    pred_idx = pred.item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = prob[0][pred_idx].item()
    print(f"Prediction: {pred_class}, Confidence: {confidence:.4f}")
    return pred_class, confidence


if __name__ == "__main__":
    # replace with your own test image path
    predict_image("./test.jpg")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from torchvision import datasets
from dataset import get_transforms
from model import MedicalImageClassifier
from torch.utils.data import DataLoader

# ==================== Config ====================
TEST_DIR = "./chest_xray/test"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model.pth"
CLASS_NAMES = ["Normal", "Pneumonia"]
# ==================================================


def build_test_loader(test_dir, batch_size=16):
    """Build test dataloader."""
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_transforms("val")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )
    return test_loader, test_dataset


@torch.no_grad()
def full_test(model, loader, device):
    """Run inference over the whole test set; return labels, preds, probs."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # positive class (Pneumonia) probability

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=200)
    print("Saved confusion_matrix.png")


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, "b-", linewidth=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./roc_curve.png", dpi=200)
    print(f"Saved roc_curve.png, AUC = {roc_auc:.4f}")


def plot_sample_predictions(dataset, y_true, y_pred, y_prob, num_samples=9):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    rows = int(np.sqrt(num_samples))
    cols = num_samples // rows

    plt.figure(figsize=(cols * 3, rows * 3.5))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img[:, :, 0], cmap="gray")
        pred_label = CLASS_NAMES[y_pred[idx]]
        true_label = CLASS_NAMES[y_true[idx]]
        conf = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
        color = "green" if y_pred[idx] == label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.3f}",
                  color=color, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("./sample_predictions.png", dpi=200)
    print("Saved sample_predictions.png")


def main():
    print(f"Device: {DEVICE}")
    test_loader, test_dataset = build_test_loader(TEST_DIR, BATCH_SIZE)
    print(f"Test samples: {len(test_dataset)}")

    model = MedicalImageClassifier(num_classes=2, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print(f"Loaded model: {MODEL_PATH}")

    y_true, y_pred, y_prob = full_test(model, test_loader, DEVICE)

    accuracy = (y_pred == y_true).mean() * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    print("\n" + "=" * 40)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=" * 40 + "\n")

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)
    plot_sample_predictions(test_dataset, y_true, y_pred, y_prob)

    print("\nDone. All charts saved to project root.")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from dataset import build_dataloader
from model import MedicalImageClassifier
# 若要切换轻量化模型，把上一行换成：
# from model_mobile import MobileNetClassifier as MedicalImageClassifier

# ==================== 配置参数 ====================
DATA_DIR = "./chest_xray"          # 改成你解压后的数据集实际路径
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
USE_CLASS_WEIGHT = True            # 是否启用类别加权损失（缓解类别不平衡）
USE_SAMPLER = False                # 是否启用过采样（True 时与加权二选一即可，避免双重放大）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "./checkpoints/best_model.pth"
os.makedirs("./checkpoints", exist_ok=True)
# ==================================================


def build_class_weights(targets, device):
    """根据训练集各类样本数，生成反比类别权重：样本越少权重越高"""
    counts = Counter(targets)
    num_classes = len(counts)
    num_total = len(targets)
    weights = torch.tensor(
        [num_total / (num_classes * counts[c]) for c in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    print(f"各类样本数: {dict(counts)}")
    print(f"类别权重: {[round(w.item(), 3) for w in weights]}")
    return weights


def build_sampler(targets, class_weights):
    """按类别权重做过采样，让两类被抽到的概率接近"""
    sample_weights = [class_weights[t].item() for t in targets]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(targets), replacement=True
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    print(f"使用设备: {DEVICE}")
    train_loader, val_loader, classes = build_dataloader(DATA_DIR, BATCH_SIZE)
    print(f"数据集类别: {classes}")

    # ---- 类别不平衡处理 ----
    targets = train_loader.dataset.targets
    class_weights = build_class_weights(targets, DEVICE) if USE_CLASS_WEIGHT else None

    if USE_SAMPLER:
        # 用采样器替换 shuffle，需要重建 train_loader
        sampler = build_sampler(targets, class_weights)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=BATCH_SIZE,
            sampler=sampler, num_workers=0, pin_memory=False
        )

    model = MedicalImageClassifier(num_classes=len(classes)).to(DEVICE)

    if USE_CLASS_WEIGHT:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ 最佳模型已更新，验证准确率: {best_acc:.2f}%")

    print(f"\n🏁 训练完成，最佳验证准确率: {best_acc:.2f}%")

    torch.save({
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "val_loss": val_loss_list,
        "val_acc": val_acc_list
    }, "./checkpoints/train_log.pth")


if __name__ == "__main__":
    main()

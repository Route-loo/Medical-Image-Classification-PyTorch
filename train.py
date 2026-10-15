import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import build_dataloader
from model import MedicalImageClassifier

# ==================== 配置参数 ====================
DATA_DIR = "./chest_xray"  # 改成你解压后的数据集实际路径
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "./checkpoints/best_model.pth"
os.makedirs("./checkpoints", exist_ok=True)


# ==================================================

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

    model = MedicalImageClassifier(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    # 记录训练曲线
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

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ 最佳模型已更新，验证准确率: {best_acc:.2f}%")

    print(f"\n🏁 训练完成，最佳验证准确率: {best_acc:.2f}%")

    # 保存训练曲线数据，方便画图
    torch.save({
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "val_loss": val_loss_list,
        "val_acc": val_acc_list
    }, "./checkpoints/train_log.pth")


if __name__ == "__main__":
    main()

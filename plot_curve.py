import torch
import matplotlib.pyplot as plt

log = torch.load("./checkpoints/train_log.pth")
epochs = range(1, len(log["train_loss"]) + 1)

plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, log["train_loss"], "b-", label="训练损失")
plt.plot(epochs, log["val_loss"], "r-", label="验证损失")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("损失曲线")
plt.legend()
plt.grid(alpha=0.3)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, log["train_acc"], "b-", label="训练准确率")
plt.plot(epochs, log["val_acc"], "r-", label="验证准确率")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("准确率曲线")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./training_curve.png", dpi=200)
print("训练曲线图已保存为 training_curve.png")

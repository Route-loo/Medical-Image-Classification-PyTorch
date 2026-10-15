from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(mode="train"):
    """医学图像数据增强：训练集做增强，验证集只做标准化"""
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Grayscale(num_output_channels=3),  # 灰度胸片转3通道适配ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def build_dataloader(data_dir, batch_size=8, num_workers=0):
    """
    适配Kaggle儿童肺炎数据集目录结构：
    data_dir/
    ├── train/
    │   ├── NORMAL/     # 正常胸片
    │   └── PNEUMONIA/  # 肺炎病灶
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/
    num_workers默认0，兼容Windows系统
    """
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=get_transforms("train")
    )
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=get_transforms("val")
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, train_dataset.classes

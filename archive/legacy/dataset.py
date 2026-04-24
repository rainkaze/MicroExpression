import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def get_dataloaders(data_dir, batch_size=128, img_size=128):
    """获取训练和验证数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 增加数据多样性
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载原始数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 打印类别映射关系，方便后续预测
    print(f"检测到的类别映射: {full_dataset.class_to_idx}")

    # 划分 80% 训练，20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Windows 下 num_workers 建议设为 0 以防多进程卡死
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    return train_loader, val_loader, full_dataset.class_to_idx
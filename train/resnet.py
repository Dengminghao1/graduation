import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. 配置参数 ---
data_dir = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\classified_frames"  # 你之前分类好的根目录
batch_size = 32
num_epochs = 20
learning_rate = 0.001
num_classes = 5  # 低, 稍低, 中性, 稍高, 高
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据增强与预处理 ---
# ResNet 标准输入是 224x224
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize((224, 224)), # 确保万一有非224的图片进入
        # transforms.CenterCrop(224),    # 仅仅是从中心截取，不进行随机变换
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 标准标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. 加载数据集并划分训练/验证集 ---
full_dataset = datasets.ImageFolder(data_dir)

# 获取索引进行划分 (80% 训练, 20% 验证)
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=full_dataset.targets,  # 保持类别比例一致
    random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
train_dataset.dataset.transform = data_transforms['train']  # 这种写法在某些版本PyTorch下需注意，推荐下面这种：


# 修正：为训练和验证创建独立的实例以应用不同的 Transform
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


train_loader = DataLoader(ApplyTransform(Subset(full_dataset, train_idx), data_transforms['train']),
                          batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ApplyTransform(Subset(full_dataset, val_idx), data_transforms['val']),
                        batch_size=batch_size, shuffle=False, num_workers=4)

# --- 4. 构建 ResNet 模型 ---
print(f"正在加载预训练 ResNet50 并运行在: {device}")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 修改最后的全连接层以匹配你的 5 分类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# --- 5. 损失函数与优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. 训练循环 ---
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')

    # 训练阶段
    model.train()
    running_loss = 0.0
    corrects = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_idx)
    epoch_acc = corrects.double() / len(train_idx)

    # 验证阶段
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

    val_acc = val_corrects.double() / len(val_idx)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val Acc: {val_acc:.4f}')

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_resnet_model.pth')

print(f'训练完成! 最佳验证准确率: {best_acc:.4f}')
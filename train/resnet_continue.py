import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# --- 1. 配置参数 (根据 24G 显存优化) ---
data_dir = r"/home/ccnu/Desktop/2021214387_周婉婷/total/classified_frames"
batch_size = 256
start_epoch = 20  # 记录从第 21 轮开始
total_epochs = 60  # 目标总轮数建议设为 60
learning_rate = 0.0001  # 续训建议学习率减小 10 倍，进行微调
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据增强 (保持不变) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. 加载数据 (保持不变) ---
full_dataset = datasets.ImageFolder(data_dir)
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2, stratify=full_dataset.targets, random_state=42
)


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
                          batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(ApplyTransform(Subset(full_dataset, val_idx), data_transforms['val']),
                        batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# --- 4. 构建模型并加载权重 (关键修改) ---
model = models.resnet50(weights=None)  # 不再需要下载官方预训练权重
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# 加载你之前训练好的权重
weight_path = 'best_resnet_model.pth'
if os.path.exists(weight_path):
    print(f"正在加载已有权重: {weight_path}")
    model.load_state_dict(torch.load(weight_path))
else:
    print("警告：未找到权重文件，将从零开始训练！")

model = model.to(device)

# --- 5. 损失函数与优化器 (增加 Scheduler) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 自动调整学习率：如果验证集 Acc 3轮不升，学习率减半
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# --- 6. 训练循环 ---
best_acc = 0.5504  # 填入你之前的最佳准确率，只有超过它才会保存新模型

for epoch in range(start_epoch, total_epochs):
    print(f'Epoch {epoch + 1}/{total_epochs}')

    model.train()
    running_loss, corrects = 0.0, 0
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

    # 更新学习率调度器
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val Acc: {val_acc:.4f} LR: {current_lr}')

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_resnet_model_v2.pth')
        print(f"检测到更高准确率，模型已更新保存。")

print(f'续训完成! 最终最佳验证准确率: {best_acc:.4f}')
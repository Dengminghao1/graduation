import glob

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import autocast
from torch.cuda.amp import GradScaler
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
# 如果是续训，建议初始化为空列表；如果你有之前的历史数据也可以在此加载
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

# 混合精度缩放器（RTX 4090 必备）
scaler = GradScaler()

# 自动调整学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# --- 2. 训练循环 ---
best_acc = 0.5504  # 初始最佳准确率
total_epochs = 15  # 举例
start_epoch = 0  # 如果从头开始是0，如果是续训需手动指定或从checkpoint读取

print(f"开始续训... 目标 Epochs: {total_epochs}, 当前最佳 Acc: {best_acc:.4f}")

for epoch in range(start_epoch, total_epochs):
    # --- 训练阶段 ---
    model.train()
    train_running_loss = 0.0
    train_corrects = 0
    train_total = 0

    # 使用 tqdm 包装并显示当前 Epoch 信息
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]")

    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 4090 混合精度前向传播
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 反向传播缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计信息
        batch_size = inputs.size(0)
        train_running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        train_corrects += torch.sum(preds == labels.data)
        train_total += batch_size

        # 更新 tqdm 右侧信息
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_train_loss = train_running_loss / train_total
    epoch_train_acc = train_corrects.double() / train_total

    # --- 验证阶段 ---
    model.eval()
    val_running_loss = 0.0
    val_corrects = 0
    val_total = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Val]")

    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                v_loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            val_running_loss += v_loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            val_total += batch_size

            val_bar.set_postfix(v_loss=f"{v_loss.item():.4f}")

    epoch_val_loss = val_running_loss / val_total
    epoch_val_acc = val_corrects.double() / val_total

    # 更新学习率调度器
    scheduler.step(epoch_val_acc)
    current_lr = optimizer.param_groups[0]['lr']

    # 记录历史数据用于绘图
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc.item())
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc.item())

    # 打印 Epoch 总结
    print(f"\n[Summary] Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | LR: {current_lr}")

    # --- 保存最佳模型 (文件名去 0.) ---
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        # 清理旧的 best 模型
        for old_file in glob.glob("best_model_acc_*.pth"):
            os.remove(old_file)

        acc_suffix = int(best_acc * 10000)
        save_path = f'best_model_acc_{acc_suffix}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"🌟 检测到更高准确率，已保存新模型: {save_path}")

# --- 3. 绘制并保存学习曲线 ---
plt.figure(figsize=(12, 5))

# Loss 图像
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Acc 图像
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.axhline(y=best_acc, color='g', linestyle='--', label='Previous Best')  # 标出续训前的基准线
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('resume_training_results.png')
plt.show()

print(f'续训完成! 最终最佳验证准确率: {best_acc:.4f}')
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.cuda.amp import GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from train.resnet_face import criterion

# 配置与之前保持一致
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_dir = r"/home/ccnu/Desktop/dataset/classified_frames_by_label_all"
batch_size = 256
num_epochs = 50
learning_rate = 0.00005  # 续训时建议使用更小的学习率，或者保持之前的
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 续训专用参数 ---
RESUME_MODEL = 'best_model_acc_8487.pth'  # 修改为你的文件名
START_EPOCH = 50  # 假设之前跑了50轮

# 1. 数据准备 (与之前一致)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

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

    def __len__(self): return len(self.subset)


train_loader = DataLoader(ApplyTransform(Subset(full_dataset, train_idx), data_transforms['train']),
                          batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ApplyTransform(Subset(full_dataset, val_idx), data_transforms['val']),
                        batch_size=batch_size, shuffle=False, num_workers=4)

# 2. 构建模型并加载权重
model = models.resnet50(weights=None)  # 续训不需要重复下载 ImageNet 权重
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))

if os.path.exists(RESUME_MODEL):
    print(f"🚀 正在从 {RESUME_MODEL} 恢复训练...")
    model.load_state_dict(torch.load(RESUME_MODEL, map_location=device))
else:
    print("❌ 未找到预训练模型，请检查路径。")
    exit()

model = model.to(device)

# 3. 优化器与调度器
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
scaler = GradScaler()

# 4. 训练循环 (加入早停和历史记录)
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = float(RESUME_MODEL.split('_')[-1].split('.')[0]) / 10000.0  # 从文件名解析出之前的 Acc
patience_counter = 0
early_stop_patience = 10

for epoch in range(START_EPOCH, START_EPOCH + num_epochs):
    # --- 1. 训练阶段 ---
    model.train()
    running_loss = 0.0
    corrects = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
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

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        total_train += inputs.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = corrects.double() / total_train

    # --- 2. 验证阶段 ---
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                v_loss = criterion(outputs, labels)

            val_loss += v_loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            total_val += inputs.size(0)

    epoch_val_loss = val_loss / total_val
    epoch_val_acc = val_corrects.double() / total_val
    scheduler.step(epoch_val_loss)
    # 记录数据用于绘图
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc.item())
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc.item())

    print(f'Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
    # --- 建议增加：早停机制 (Early Stopping) ---
    # 防止后面 20 个 epoch 都在浪费电并加剧过拟合
    # --- 3. 保存最佳模型 (文件名不要 0.) ---
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_counter = 0  # 重置计数器
        # 清除旧的 best 模型
        for old_file in glob.glob("best_model_acc_*.pth"):
            os.remove(old_file)

        # 转换准确率为整数，如 0.9542 -> 9542
        acc_suffix = int(best_val_acc * 10000)
        save_path = f'best_model_acc_{acc_suffix}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"🌟 发现更优模型: {save_path}")

    else:
        patience_counter += 1
        print(f"⚠ 验证集表现未提升，早停计数器: {patience_counter}/{early_stop_patience}")

    # 触发早停
    if patience_counter >= early_stop_patience:
        print("🛑 [Early Stopping] 验证集表现长期停滞，提前结束训练。")
        break
    # --- 4. 绘制并保存图像 ---
plt.figure(figsize=(12, 5))

# 绘制 Loss 子图
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Val Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# 绘制 Accuracy 子图
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', color='blue')
plt.plot(history['val_acc'], label='Val Acc', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')  # 保存为图片文件
plt.show()

print(f'训练完成! 最佳验证准确率: {best_val_acc:.4f}')

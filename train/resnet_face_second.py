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

# 用第二块显卡训练
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# --- 1. 配置参数 ---
data_dir = r"D:\dataset\frame_picture\classified_frames_face_101"  # 你之前分类好的根目录
batch_size = 256
num_epochs = 100
learning_rate = 0.0001
num_classes = 5  # 低, 稍低, 中性, 稍高, 高
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据增强与预处理 ---
# ResNet 标准输入是 224x224
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

# --- 3. 加载数据集并划分训练/验证集 ---
# 自定义数据集类，支持按时间区间分组选择一帧
class TimeIntervalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.targets = []
        
        # 遍历所有类别文件夹
        class_to_idx = {}
        for i, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_to_idx[class_name] = i
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # 按时间区间分组文件
            interval_groups = {}
            for filename in os.listdir(class_path):
                if filename.endswith('.jpg'):
                    # 提取时间区间：frame_000000_192.168.0.101_01_20231229153000_20231229154000.jpg
                    parts = filename.split('_')
                    if len(parts) >= 5:
                        interval = f"{parts[-2]}_{parts[-1].split('.')[0]}"
                        if interval not in interval_groups:
                            interval_groups[interval] = []
                        interval_groups[interval].append(filename)
            
            # 每个时间区间每十张图片选取一张
            for interval, files in interval_groups.items():
                if files:
                    # 按帧号排序
                    files.sort()
                    # 每十张选取一张（均匀采样）
                    step = 10
                    for i in range(0, len(files), step):
                        # 取每十张的中间位置（第5张，索引为4）
                        selected_idx = min(i + 4, len(files) - 1)
                        selected_file = files[selected_idx]

                        img_path = os.path.join(class_path, selected_file)
                        self.samples.append(img_path)
                        self.targets.append(class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)

# 创建数据集实例

full_dataset = TimeIntervalDataset(data_dir)

# 获取索引进行划分 (80% 训练, 20% 验证)
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=full_dataset.targets,  # 保持类别比例一致
    random_state=42
)

# 修正：为训练和验证创建独立的实例以应用不同的 Transform
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        img_path, target = self.dataset.samples[self.indices[index]], self.dataset.targets[self.indices[index]]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)

train_loader = DataLoader(ApplyTransform(full_dataset, train_idx, data_transforms['train']),
                          batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ApplyTransform(full_dataset, val_idx, data_transforms['val']),
                        batch_size=batch_size, shuffle=False, num_workers=4)

# --- 4. 构建 ResNet 模型 ---
print(f"正在加载预训练 ResNet50 并运行在: {device}")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 修改最后的全连接层以匹配你的 5 分类
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)
model = model.to(device)

# --- 5. 损失函数与优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# 4. 增加学习率调整策略
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
# --- 6. 训练循环 ---
# 初始化用于记录绘图数据的字典
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0
scaler = GradScaler()  # 4090 混合精度加速器

print(f"开始训练... 设备: {device}")

patience_counter = 0
early_stop_patience = 10
for epoch in range(num_epochs):
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

        # 清除旧的 best 模型（只删除准确率低于当前最佳的）
        for old_file in glob.glob("best_model_acc_face_lstm_*.pth"):
            # 从文件名中提取准确率
            try:
                old_acc_str = old_file.split('_')[-1].split('.')[0]
                old_acc = int(old_acc_str) / 10000
                # 只有当旧模型的准确率低于当前最佳准确率时才删除
                if old_acc < best_val_acc:
                    os.remove(old_file)
                    print(f"🔄 删除旧模型: {old_file} (准确率: {old_acc:.4f})")
            except:
                # 如果文件名格式不正确，也删除
                os.remove(old_file)
                print(f"🔄 删除格式不正确的旧模型: {old_file}")

        # 保存新模型
        acc_suffix = int(best_val_acc * 10000)
        save_path = f'best_model_acc_face_{acc_suffix}.pth'
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
plt.savefig('training_results_face.png')  # 保存为图片文件
plt.show()

print(f'训练完成! 最佳验证准确率: {best_val_acc:.4f}')
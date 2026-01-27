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
face_data_dir = r"/home/ccnu/Desktop/dataset/classified_frames_face_by_label_all"  # 面部数据
pose_data_dir = r"/home/ccnu/Desktop/dataset/classified_frames_pose_by_label_all"  # 肢体数据
batch_size = 128  # 减半以适应双输入
num_epochs = 100
learning_rate = 0.0001
num_classes = 5  # 低, 稍低, 中性, 稍高, 高
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据增强与预处理 ---
# ResNet 标准输入是 224x224
# 使用原有参数：面部和肢体分别使用各自的标准化参数
data_transforms = {
    'face_train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'face_val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'pose_train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.98, 1.02)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]),
    'pose_val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]),
}

# --- 3. 自定义数据集加载器 --- 
class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, face_subset, pose_subset, face_transform=None, pose_transform=None):
        self.face_subset = face_subset
        self.pose_subset = pose_subset
        self.face_transform = face_transform
        self.pose_transform = pose_transform
    
    def __getitem__(self, index):
        # 获取面部图像和标签
        face_img, label = self.face_subset[index]
        # 获取对应索引的肢体图像
        pose_img, _ = self.pose_subset[index]
        
        if self.face_transform:
            face_img = self.face_transform(face_img)
        if self.pose_transform:
            pose_img = self.pose_transform(pose_img)
        
        return face_img, pose_img, label
    
    def __len__(self):
        return len(self.face_subset)

# --- 4. 加载数据集并划分训练/验证集 ---
print("正在加载面部和肢体数据集...")
face_full_dataset = datasets.ImageFolder(face_data_dir)
pose_full_dataset = datasets.ImageFolder(pose_data_dir)

# 获取索引进行划分 (80% 训练, 20% 验证)
train_idx, val_idx = train_test_split(
    list(range(len(face_full_dataset))),
    test_size=0.2,
    stratify=face_full_dataset.targets,  # 保持类别比例一致
    random_state=42
)

# 创建训练和验证数据集
train_dataset = FusionDataset(
    Subset(face_full_dataset, train_idx),
    Subset(pose_full_dataset, train_idx),
    face_transform=data_transforms['face_train'],
    pose_transform=data_transforms['pose_train']
)
val_dataset = FusionDataset(
    Subset(face_full_dataset, val_idx),
    Subset(pose_full_dataset, val_idx),
    face_transform=data_transforms['face_val'],
    pose_transform=data_transforms['pose_val']
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- 5. 构建融合模型 --- 
class FusionResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(FusionResNet, self).__init__()
        
        # 面部分支
        self.face_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.face_backbone.fc = nn.Identity()
        
        # 肢体分支
        self.pose_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.pose_backbone.fc = nn.Identity()
        
        # 获取特征维度
        self.feature_dim = self.face_backbone.fc.in_features
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim * 2, num_classes)
        )
    
    def forward(self, face_x, pose_x):
        # 提取特征
        face_feat = self.face_backbone(face_x)
        pose_feat = self.pose_backbone(pose_x)
        
        # 特征融合
        combined = torch.cat([face_feat, pose_feat], dim=1)
        
        # 注意力加权
        attention_weights = self.attention(combined)
        face_attn = attention_weights[:, 0].unsqueeze(1) * face_feat
        pose_attn = attention_weights[:, 1].unsqueeze(1) * pose_feat
        
        # 加权融合
        fused = torch.cat([face_attn, pose_attn], dim=1)
        
        # 分类
        output = self.classifier(fused)
        
        return output

print(f"正在加载融合模型并运行在: {device}")
model = FusionResNet(num_classes=num_classes)
model = model.to(device)

# --- 6. 损失函数与优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# --- 7. 训练循环 ---
# 初始化用于记录绘图数据的字典
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0
scaler = GradScaler()  # 混合精度加速器

print(f"开始训练... 设备: {device}")

patience_counter = 0
early_stop_patience = 10

for epoch in range(num_epochs):
    # --- 1. 训练阶段 ---
    model.train()
    running_loss = 0.0
    corrects = 0
    total_train = 0

    for face_inputs, pose_inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
        face_inputs, pose_inputs, labels = face_inputs.to(device), pose_inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(device_type='cuda'):
            outputs = model(face_inputs, pose_inputs)
            loss = criterion(outputs, labels)

        # 反向传播缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计
        running_loss += loss.item() * face_inputs.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        total_train += face_inputs.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = corrects.double() / total_train

    # --- 2. 验证阶段 ---
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    total_val = 0

    with torch.no_grad():
        for face_inputs, pose_inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
            face_inputs, pose_inputs, labels = face_inputs.to(device), pose_inputs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(face_inputs, pose_inputs)
                v_loss = criterion(outputs, labels)

            val_loss += v_loss.item() * face_inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            total_val += face_inputs.size(0)

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
    
    # --- 保存最佳模型 ---
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_counter = 0  # 重置计数器

        # 转换准确率为整数，如 0.9542 -> 9542
        acc_suffix = int(best_val_acc * 10000)
        save_path = f'best_fusion_model_acc_{acc_suffix}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"🌟 发现更优模型: {save_path}")

    else:
        patience_counter += 1
        print(f"⚠ 验证集表现未提升，早停计数器: {patience_counter}/{early_stop_patience}")

    # 触发早停
    if patience_counter >= early_stop_patience:
        print("🛑 [Early Stopping] 验证集表现长期停滞，提前结束训练。")
        break

# --- 绘制并保存图像 ---
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
plt.savefig('fusion_training_results.png')  # 保存为图片文件
plt.show()

print(f'训练完成! 最佳验证准确率: {best_val_acc:.4f}')

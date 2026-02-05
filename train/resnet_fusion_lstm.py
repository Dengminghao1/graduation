import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch import autocast
from torch.cuda.amp import GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# 用第二块显卡训练
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# --- 1. 配置参数 ---
face_data_dir = r"/home/ccnu/Desktop/dataset/extracted_frames_pic/face_extracted_frames_all"  # 面部数据
pose_data_dir = r"/home/ccnu/Desktop/dataset/extracted_frames_pic/pose_extracted_frames_all"  # 肢体数据
csv_dir = r"/home/ccnu/Desktop/dataset/eeg_csv"  # 标签CSV文件目录

# face_data_dir = r"D:\dataset\frame_picture\face_extracted_frames_101"  # 面部数据
# pose_data_dir = r"D:\dataset\frame_picture\pose_extracted_frames_101"  # 肢体数据
# csv_dir = r"D:\dataset\eeg_csv"  # 标签CSV文件目录

batch_size = 32  # 进一步减小以适应序列输入
num_epochs = 100
learning_rate = 0.0001
num_classes = 5  # 低, 稍低, 中性, 稍高, 高
sequence_length = 10  # 帧序列长度（窗口大小为十）
hidden_size = 512  # LSTM 隐藏层大小
num_layers = 2  # LSTM 层数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- MultiSegmentAttentionDataset 类 --- 
class MultiSegmentAttentionDataset(Dataset):
    def __init__(self, img_dir, csv_dir, seq_len=10, transform=None, segment_keys=None, is_pose=False):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.label_map = {'低': 0, '稍低': 1, '中性': 2, '稍高': 3, '高': 4}
        self.is_pose = is_pose

        # 1. 扫描并加载 CSV 标签库
        # key: (start_time_str, end_time_str), value: DataFrame
        self.label_dfs = {}
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        print(f"正在加载 {len(csv_files)} 个标签文件...")

        for cf in csv_files:
            # 假设 CSV 文件名格式: xxxx_xxxx_20231229153000_20231229154000.csv
            parts = cf.replace('.csv', '').split('_')
            # 根据你的文件名规则，倒数第二和倒数第一通常是时间
            s_str, e_str = parts[-2], parts[-1]

            df = pd.read_csv(os.path.join(csv_dir, cf))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.label_dfs[(s_str, e_str)] = df

        # 2. 解析图像文件并按时间段分组
        segments = defaultdict(list)
        if self.is_pose:
            all_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        else:
            all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        print(f"正在解析 {len(all_files)} 个图像文件...")

        for f in all_files:
            parts = f.split('_')
            if self.is_pose:
                # 肢体格式：192.168.0.101_01_20231229153000_20231229154000_000000002190_rendered.png
                if len(parts) >= 6:
                    frame_idx = int(parts[-2])
                    s_time_str = parts[-4]
                    e_time_str = parts[-3]
            else:
                # 面部格式：frame_000000_192.168.0.101_01_20231229153000_20231229154000.jpg
                if len(parts) >= 5:
                    frame_idx = int(parts[1])
                    s_time_str = parts[-2]
                    e_time_str = parts[-1].replace('.jpg', '')

            start_dt = datetime.strptime(s_time_str, "%Y%m%d%H%M%S")
            total_milliseconds = frame_idx * 100  # 0.1秒 = 100毫秒
            curr_dt = start_dt + timedelta(milliseconds=total_milliseconds)
            segments[(s_time_str, e_time_str)].append({
                'filename': f,
                'time': curr_dt,
                'idx': frame_idx
            })

        # 3. 如果指定了 segment_keys，则只保留这些段的数据
        if segment_keys is not None:
            filtered_segments = {k: v for k, v in segments.items() if k in segment_keys}
        else:
            filtered_segments = segments

        # 4. 构建序列并从"对应"的 DataFrame 中取标签
        self.valid_sequences = []
        print("开始时序匹配标签...")

        for seg_key, seg_files_list in filtered_segments.items():
            # 检查是否有对应的 CSV 标签
            if seg_key not in self.label_dfs:
                print(f"⚠ 警告: 未找到段 {seg_key} 对应的 CSV 标签，跳过...")
                continue

            current_df = self.label_dfs[seg_key]
            seg_files = sorted(seg_files_list, key=lambda x: x['idx'])

            for i in range(0, len(seg_files) - seq_len + 1, 10):
                seq_frames = seg_files[i: i + seq_len]
                end_frame_time = seq_frames[-1]['time']

                # 使用最后一帧的标签
                end_frame_time = seq_frames[-1]['time']
                time_diffs = (current_df['timestamp'] - end_frame_time).abs()
                closest_idx = time_diffs.idxmin()
                label_str = current_df.loc[closest_idx, 'attention']
                
                self.valid_sequences.append({
                    'files': [x['filename'] for x in seq_frames],
                    'label': self.label_map.get(label_str, 2)
                })

        print(f"成功构建序列总数: {len(self.valid_sequences)}")

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        data = self.valid_sequences[idx]
        frames = []
        for fname in data['files']:
            img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                # 如果没有transform，至少转换为tensor
                from torchvision import transforms
                img = transforms.ToTensor()(img)
            frames.append(img)
        # 返回帧序列和标签
        return torch.stack(frames), torch.tensor(data['label'], dtype=torch.long)



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

# --- 3. 自定义数据集类 ---
# 应用变换的数据集类
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

# --- 4. 自定义数据集加载器 --- 
class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, face_data_dir, pose_data_dir, csv_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []
        self.targets = []
        
        # 加载面部和肢体数据
        face_dataset = MultiSegmentAttentionDataset(face_data_dir, csv_dir, sequence_length, is_pose=False)
        pose_dataset = MultiSegmentAttentionDataset(pose_data_dir, csv_dir, sequence_length, is_pose=True)
        
        # 按序匹配面部和肢体样本
        matched_count = 0
        
        # 遍历面部序列
        for face_seq in face_dataset.valid_sequences:
            # 获取面部序列的时间区间和帧号
            first_face_img = face_seq['files'][0]
            face_parts = first_face_img.split('_')
            if len(face_parts) >= 5:
                # 面部格式：frame_000000_192.168.0.101_01_20231229153000_20231229154000.jpg
                face_interval = f"{face_parts[-2]}_{face_parts[-1].split('.')[0]}"
                face_frame_num = int(face_parts[1])
                
                # 在肢体序列中查找匹配的序列
                for pose_seq in pose_dataset.valid_sequences:
                    first_pose_img = pose_seq['files'][0]
                    pose_parts = first_pose_img.split('_')
                    if len(pose_parts) >= 6:
                        # 肢体格式：192.168.0.101_01_20231229153000_20231229154000_000000002190_rendered.png
                        pose_interval = f"{pose_parts[-4]}_{pose_parts[-3]}"
                        pose_frame_num = int(pose_parts[-2])
                        
                        # 检查时间区间和帧号是否匹配
                        if face_interval == pose_interval and abs(face_frame_num - pose_frame_num) == 0:
                            # 确保序列长度一致
                            if len(face_seq['files']) == sequence_length and len(pose_seq['files']) == sequence_length:
                                self.samples.append((face_seq['files'], pose_seq['files']))
                                self.targets.append(face_seq['label'])
                                matched_count += 1
                                break
        
        print(f"成功匹配 {matched_count} 对序列样本")
    
    def __getitem__(self, index):
        face_img_paths, pose_img_paths = self.samples[index]
        target = self.targets[index]
        
        from PIL import Image
        face_imgs = []
        pose_imgs = []
        
        # 加载序列中的所有图像
        for face_path, pose_path in zip(face_img_paths, pose_img_paths):
            face_img = Image.open(face_path).convert('RGB')
            pose_img = Image.open(pose_path).convert('RGB')
            face_imgs.append(face_img)
            pose_imgs.append(pose_img)
        
        return face_imgs, pose_imgs, target
    
    def __len__(self):
        return len(self.samples)

# 应用变换的融合数据集类
class FusionApplyTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, face_transform=None, pose_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.face_transform = face_transform
        self.pose_transform = pose_transform

    def __getitem__(self, index):
        face_imgs, pose_imgs, target = self.dataset[self.indices[index]]
        transformed_face_imgs = []
        transformed_pose_imgs = []
        
        # 对序列中的每个图像应用变换
        for face_img, pose_img in zip(face_imgs, pose_imgs):
            if self.face_transform:
                face_img = self.face_transform(face_img)
            if self.pose_transform:
                pose_img = self.pose_transform(pose_img)
            transformed_face_imgs.append(face_img)
            transformed_pose_imgs.append(pose_img)
        
        # 将列表转换为张量，维度为 (序列长度, 通道, 高度, 宽度)
        transformed_face_imgs = torch.stack(transformed_face_imgs)
        transformed_pose_imgs = torch.stack(transformed_pose_imgs)
        
        return transformed_face_imgs, transformed_pose_imgs, target

    def __len__(self):
        return len(self.indices)

# --- 5. 加载数据集并划分训练/验证集 ---  
print("正在加载面部和肢体数据集...")

# 创建完整的融合数据集（已匹配的样本）
full_dataset = FusionDataset(face_data_dir, pose_data_dir, csv_dir, sequence_length=sequence_length)

# 获取索引进行划分 (80% 训练, 20% 验证)
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=full_dataset.targets,  # 保持类别比例一致
    random_state=42
)

# 创建训练和验证数据集
train_dataset = FusionApplyTransform(
    full_dataset,
    train_idx,
    face_transform=data_transforms['face_train'],
    pose_transform=data_transforms['pose_train']
)
val_dataset = FusionApplyTransform(
    full_dataset,
    val_idx,
    face_transform=data_transforms['face_val'],
    pose_transform=data_transforms['pose_val']
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- 5. 构建融合模型 --- 
class FusionResNetLSTM(nn.Module):
    def __init__(self, num_classes=5, sequence_length=5, hidden_size=512, num_layers=2):
        super(FusionResNetLSTM, self).__init__()
        
        # 面部分支
        self.face_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 保存原始 fc 层的 in_features
        self.feature_dim = self.face_backbone.fc.in_features
        self.face_backbone.fc = nn.Identity()
        
        # 肢体分支
        self.pose_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.pose_backbone.fc = nn.Identity()
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
        # LSTM 层
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入到 LSTM 的特征维度是融合后的特征维度
        self.lstm = nn.LSTM(
            input_size=self.feature_dim * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, face_x, pose_x):
        # 输入维度: (batch_size, sequence_length, channels, height, width)
        batch_size = face_x.size(0)
        sequence_length = face_x.size(1)
        
        # 调整维度以适应 ResNet: (batch_size * sequence_length, channels, height, width)
        face_x_reshaped = face_x.view(-1, face_x.size(2), face_x.size(3), face_x.size(4))
        pose_x_reshaped = pose_x.view(-1, pose_x.size(2), pose_x.size(3), pose_x.size(4))
        
        # 提取特征
        face_feat = self.face_backbone(face_x_reshaped)
        pose_feat = self.pose_backbone(pose_x_reshaped)
        
        # 调整特征维度: (batch_size, sequence_length, feature_dim)
        face_feat = face_feat.view(batch_size, sequence_length, -1)
        pose_feat = pose_feat.view(batch_size, sequence_length, -1)
        
        # 特征融合与注意力加权
        fused_features = []
        for t in range(sequence_length):
            # 获取当前时间步的特征
            face_feat_t = face_feat[:, t, :]
            pose_feat_t = pose_feat[:, t, :]
            
            # 特征融合
            combined = torch.cat([face_feat_t, pose_feat_t], dim=1)
            
            # 注意力加权
            attention_weights = self.attention(combined)
            face_attn = attention_weights[:, 0].unsqueeze(1) * face_feat_t
            pose_attn = attention_weights[:, 1].unsqueeze(1) * pose_feat_t
            
            # 加权融合
            fused = torch.cat([face_attn, pose_attn], dim=1)
            fused_features.append(fused.unsqueeze(1))
        
        # 堆叠所有时间步的融合特征: (batch_size, sequence_length, feature_dim * 2)
        fused_sequence = torch.cat(fused_features, dim=1)
        
        # LSTM 处理
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(face_x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(face_x.device)
        
        # 前向传播通过 LSTM
        lstm_out, (hn, cn) = self.lstm(fused_sequence, (h0, c0))
        
        # 使用最后一个时间步的输出进行分类
        output = self.classifier(hn[-1])
        
        return output

print(f"正在加载融合模型并运行在: {device}")
model = FusionResNetLSTM(
    num_classes=num_classes,
    sequence_length=sequence_length,
    hidden_size=hidden_size,
    num_layers=num_layers
)
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

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_counter = 0  # 重置计数器

        # 清除旧的 best 模型（只删除准确率低于当前最佳的）
        for old_file in glob.glob("best_model_acc_fusion_lstm_*.pth"):
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
        save_path = f'best_model_acc_fusion_lstm_{acc_suffix}.pth'
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
plt.savefig('fusion_lstm_training_results.png')  # 保存为图片文件
plt.show()

print(f'训练完成! 最佳验证准确率: {best_val_acc:.4f}')

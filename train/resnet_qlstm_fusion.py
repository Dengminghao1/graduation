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
import pandas as pd
from datetime import datetime, timedelta
# 用第二块显卡训练
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# --- 1. 配置参数 ---
face_data_dir = r"/home/ccnu/Desktop/dataset/classified_frames_face_by_label_all"  # 面部数据
pose_data_dir = r"/home/ccnu/Desktop/dataset/classified_frames_pose_by_label_all"  # 肢体数据
# 添加置信度CSV文件目录参数
CONF_CSV_PATH = r"/home/ccnu/Desktop/dataset/confidence.csv"  # 置信度数据
batch_size = 32  # 进一步减小以适应序列输入
num_epochs = 100
learning_rate = 0.0001
num_classes = 5  # 低, 稍低, 中性, 稍高, 高
sequence_length = 10  # 帧序列长度（窗口大小为十）
hidden_size = 512  # LSTM 隐藏层大小
num_layers = 2  # LSTM 层数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Quality-aware Temporal Attention LSTM
class QTALSTMCell(nn.Module):
    """
    Quality-aware Temporal Attention LSTM Cell (Confidence-only Version)
    """

    def __init__(self, input_size, hidden_size, use_smooth=True):
        super(QTALSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_smooth = use_smooth

        # LSTM gates
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        # Learnable decay weight
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Adaptive decay strength
        self.lambda_param = nn.Parameter(torch.tensor(0.5))

        # For EMA smoothing
        self.register_buffer("conf_prev", torch.zeros(1))


    def forward(self, x_t, h_prev, c_prev, conf_t):
        """
        x_t     : [B, D]
        h_prev  : [B, H]
        c_prev  : [B, H]
        conf_t  : [B]   in [0,1]
        """

        # ---------- 1. Confidence smoothing (optional) ----------
        if self.use_smooth:
            if self.conf_prev.numel() != conf_t.numel():
                self.conf_prev = conf_t.detach()

            conf_s = 0.8 * self.conf_prev + 0.2 * conf_t
            self.conf_prev = conf_s.detach()
        else:
            conf_s = conf_t

        # ---------- 2. Quality-aware delta ----------
        alpha = torch.relu(self.alpha)

        delta_t = alpha * (1 - conf_s)

        # ---------- 3. Exponential decay ----------
        s_t = torch.exp(-delta_t).unsqueeze(1)

        # ---------- 4. Adaptive memory attenuation ----------
        lambda_ = torch.sigmoid(self.lambda_param)

        decay = 1 - lambda_ * (1 - s_t)

        c_prev = c_prev * decay

        # ---------- 5. LSTM gates ----------
        combined = torch.cat([x_t, h_prev], dim=1)

        gates = self.W(combined)

        f_t, i_t, o_t, g_t = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f_t)
        i_t = torch.sigmoid(i_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        # ---------- 6. Input modulation ----------
        i_t = i_t * s_t

        # ---------- 7. Cell update ----------
        c_t = f_t * c_prev + i_t * g_t

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class QTALSTM(nn.Module):

    def __init__(self, input_size, hidden_size, use_smooth=True):
        super(QTALSTM, self).__init__()

        self.hidden_size = hidden_size

        self.cell = QTALSTMCell(
            input_size,
            hidden_size,
            use_smooth
        )


    def forward(self, x, conf):
        """
        x    : [B, T, D]
        conf : [B, T]
        """

        B, T, _ = x.size()

        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)

        outputs = []

        for t in range(T):

            h, c = self.cell(
                x[:, t],
                h,
                c,
                conf[:, t]
            )

            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)



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
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # 提取时间区间：frame_000000_192.168.0.101_01_20231229153000_20231229154000.jpg 或 192.168.0.101_01_20231229153000_20231229154000_000000002190_rendered.png
                    parts = filename.split('_')
                    if len(parts) >= 5:
                        # 处理不同格式的文件名
                        if filename.endswith('.jpg'):
                            # 面部格式：frame_000000_192.168.0.101_01_20231229153000_20231229154000.jpg
                            interval = f"{parts[-2]}_{parts[-1].split('.')[0]}"
                        else:
                            # 肢体格式：192.168.0.101_01_20231229153000_20231229154000_000000002190_rendered.png
                            # 找到时间区间部分（倒数第4和倒数第3部分）
                            if len(parts) >= 6:
                                interval = f"{parts[-4]}_{parts[-3]}"
                            else:
                                continue
                        if interval not in interval_groups:
                            interval_groups[interval] = []
                        interval_groups[interval].append(filename)
            
            # 保留每个时间区间的所有图片
            for interval, files in interval_groups.items():
                if files:
                    # 按帧号排序
                    def get_frame_number(filename):
                        if filename.endswith('.jpg'):
                            # 面部格式：frame_000070_192.168.0.101_01_20231229164011_20231229165010.jpg
                            parts = filename.split('_')
                            if len(parts) >= 2 and parts[0] == 'frame':
                                try:
                                    return int(parts[1])
                                except:
                                    return 0
                        else:
                            # 肢体格式：192.168.0.101_01_20231229153000_20231229154000_000000002190_rendered.png
                            parts = filename.split('_')
                            if len(parts) >= 2:
                                try:
                                    # 提取倒数第二部分的数字
                                    return int(parts[-2])
                                except:
                                    return 0
                        return 0
                    
                    files.sort(key=get_frame_number)
                    # 保留所有图片
                    for file in files:
                        img_path = os.path.join(class_path, file)
                        self.samples.append(img_path)
                        self.targets.append(class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)

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
    def __init__(self, face_data_dir, pose_data_dir, conf_csv_path=None, sequence_length=5):
        self.sequence_length = sequence_length
        self.samples = []
        self.targets = []
        
        # 加载面部和肢体数据
        face_dataset = TimeIntervalDataset(face_data_dir)
        pose_dataset = TimeIntervalDataset(pose_data_dir)
        
        # 加载置信度 CSV 文件（如果提供）
        self.conf_df = None
        if conf_csv_path:
            print(f"正在加载置信度文件: {conf_csv_path}...")
            try:
                # 直接加载单个置信度CSV文件
                self.conf_df = pd.read_csv(conf_csv_path)
                self.conf_df['timestamp'] = pd.to_datetime(self.conf_df['timestamp_pose'])
                print(f"成功加载置信度文件，共 {len(self.conf_df)} 条记录")
            except Exception as e:
                print(f"警告: 加载置信度文件失败: {e}")
        
        # 构建面部样本的映射：{时间区间: {帧号: (路径, 标签)}}
        face_map = {}
        for img_path, target in zip(face_dataset.samples, face_dataset.targets):
            filename = os.path.basename(img_path)
            if filename.endswith('.jpg'):
                parts = filename.split('_')
                if len(parts) >= 5:
                    interval = f"{parts[-2]}_{parts[-1].split('.')[0]}"
                    frame_num = 0
                    if len(parts) >= 2 and parts[0] == 'frame':
                        try:
                            frame_num = int(parts[1])
                        except:
                            pass
                    if interval not in face_map:
                        face_map[interval] = {}
                    face_map[interval][frame_num] = (img_path, target)
        
        # 构建肢体样本的映射：{时间区间: {帧号: 路径}}
        pose_map = {}
        for img_path in pose_dataset.samples:
            filename = os.path.basename(img_path)
            if filename.endswith('.png'):
                parts = filename.split('_')
                if len(parts) >= 6:
                    interval = f"{parts[-4]}_{parts[-3]}"
                    frame_num = 0
                    try:
                        frame_num = int(parts[-2])
                    except:
                        pass
                    if interval not in pose_map:
                        pose_map[interval] = {}
                    pose_map[interval][frame_num] = img_path
        
        # 匹配面部和肢体样本并生成序列
        matched_count = 0
        for interval in face_map:
            if interval in pose_map:
                # 获取该时间区间内所有匹配的帧号
                common_frame_nums = sorted(list(set(face_map[interval].keys()) & set(pose_map[interval].keys())))
                
                # 生成连续的帧序列，窗口大小为 sequence_length，步长为 sequence_length
                for i in range(0, len(common_frame_nums) - sequence_length + 1, sequence_length):
                    # 直接取连续的窗口，不需要额外检查
                    frame_sequence = common_frame_nums[i:i+sequence_length]
                    face_sequence = []
                    pose_sequence = []
                    conf_sequence = []
                    target = None
                    valid_sequence = True
                    
                    # 收集序列中的所有帧
                    for frame_num in frame_sequence:
                        face_path, target = face_map[interval][frame_num]
                        pose_path = pose_map[interval][frame_num]
                        face_sequence.append(face_path)
                        pose_sequence.append(pose_path)
                        
                        # 生成帧的时间戳以匹配置信度
                        if self.conf_df is not None:
                            # 从文件名中提取时间信息
                            face_filename = os.path.basename(face_path)
                            parts = face_filename.split('_')
                            if len(parts) >= 6:
                                # 假设时间格式为 20231229153000
                                time_str = parts[-2]
                                try:
                                    # 解析时间
                                    frame_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
                                    # 添加帧偏移（假设每帧0.1秒）
                                    frame_time += timedelta(seconds=frame_num * 0.1)
                                    
                                    # 在置信度 DataFrame 中找最接近的时间点
                                    time_diffs = (self.conf_df['timestamp'] - frame_time).abs()
                                    closest_idx = time_diffs.idxmin()
                                    
                                    if time_diffs.min() <= timedelta(seconds=1):
                                        # 尝试获取 confidence 值，如果不存在则设为默认值 0.5
                                        confidence = self.conf_df.loc[closest_idx].get('confidence', 0.5)
                                        conf_sequence.append(confidence)
                                    else:
                                        # 如果找不到对应的时间点，使用默认值 0.5
                                        conf_sequence.append(0.5)
                                except Exception as e:
                                    # 如果解析失败，使用默认值 0.5
                                    conf_sequence.append(0.5)
                            else:
                                # 如果文件名格式不正确，使用默认值 0.5
                                conf_sequence.append(0.5)
                        else:
                            # 如果没有提供置信度文件，使用默认值 0.5
                            conf_sequence.append(0.5)
                    
                    if target is not None and valid_sequence:
                        self.samples.append((face_sequence, pose_sequence, conf_sequence))
                        self.targets.append(target)
                        matched_count += 1
        
        print(f"成功匹配 {matched_count} 对序列样本")
    
    def __getitem__(self, index):
        face_img_paths, pose_img_paths, conf_sequence = self.samples[index]
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
        
        return face_imgs, pose_imgs, conf_sequence, target
    
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
        face_imgs, pose_imgs, conf_sequence, target = self.dataset[self.indices[index]]
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
        
        # 将置信度序列转换为张量
        conf_tensor = torch.tensor(conf_sequence, dtype=torch.float32)
        
        return transformed_face_imgs, transformed_pose_imgs, conf_tensor, target

    def __len__(self):
        return len(self.indices)

# --- 5. 加载数据集并划分训练/验证集 ---  
print("正在加载面部和肢体数据集...")

# 创建完整的融合数据集（已匹配的样本）
full_dataset = FusionDataset(face_data_dir, pose_data_dir, CONF_CSV_PATH, sequence_length=sequence_length)

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
        
        # QTALSTM 层
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入到 QTALSTM 的特征维度是融合后的特征维度
        self.qlstm = QTALSTM(
            input_size=self.feature_dim * 2,
            hidden_size=hidden_size,
            use_smooth=True
        )
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, face_x, pose_x, conf):
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
        
        # QTALSTM 处理，使用外部传递的置信度
        qlstm_out = self.qlstm(fused_sequence, conf)
        
        # 使用最后一个时间步的输出进行分类
        output = self.classifier(qlstm_out[:, -1, :])
        
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

    for face_inputs, pose_inputs, conf, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
        face_inputs, pose_inputs, conf, labels = face_inputs.to(device), pose_inputs.to(device), conf.to(device), labels.to(device)
        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(device_type='cuda'):
            outputs = model(face_inputs, pose_inputs, conf)
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
        for face_inputs, pose_inputs, conf, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
            face_inputs, pose_inputs, conf, labels = face_inputs.to(device), pose_inputs.to(device), conf.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(face_inputs, pose_inputs, conf)
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
        for old_file in glob.glob("best_model_acc_fusion_qlstm_*.pth"):
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
        save_path = f'best_model_acc_fusion_qlstm_{acc_suffix}.pth'
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
plt.savefig('fusion_qlstm_training_results.png')  # 保存为图片文件
plt.show()

print(f'训练完成! 最佳验证准确率: {best_val_acc:.4f}')

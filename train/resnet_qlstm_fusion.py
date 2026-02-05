import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch import autocast
from torch.cuda.amp import GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
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

# 添加置信度CSV文件目录参数
CONF_CSV_PATH = r"/home/ccnu/Desktop/dataset/Dataset_align_face_pose_eeg_feature.csv"  # 置信度数据
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
    def __init__(self, face_data_dir, pose_data_dir, csv_dir, conf_csv_path=None, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []
        self.targets = []
        
        # 加载面部和肢体数据
        face_dataset = MultiSegmentAttentionDataset(face_data_dir, csv_dir, sequence_length, is_pose=False)
        pose_dataset = MultiSegmentAttentionDataset(pose_data_dir, csv_dir, sequence_length, is_pose=True)
        
        # 加载置信度 CSV 文件（如果提供）
        self.conf_df = None
        if conf_csv_path:
            print(f"正在加载置信度文件: {conf_csv_path}...")
            try:
                # 直接加载单个置信度CSV文件
                self.conf_df = pd.read_csv(conf_csv_path,usecols=["timestamp_pose", "confidence"])
                self.conf_df['timestamp'] = pd.to_datetime(self.conf_df['timestamp_pose'])
                print(f"成功加载置信度文件，共 {len(self.conf_df)} 条记录")
            except Exception as e:
                print(f"警告: 加载置信度文件失败: {e}")
        
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
                                # 生成置信度序列
                                conf_sequence = []
                                
                                # 从 face_interval 中提取开始时间
                                parts = first_face_img.split('_')
                                if len(parts) >= 6:
                                    # face_interval 的开始时间是 parts[-2]
                                    start_time_str = face_parts[-2]
                                    
                                    try:
                                        # 解析开始时间
                                        start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")
                                        
                                        # 为序列中的每个帧生成置信度
                                        for i in range(sequence_length):
                                            # 计算当前帧的帧号
                                            frame_num = face_frame_num + i
                                            # 添加帧偏移（假设每帧0.1秒）
                                            frame_time = start_time + timedelta(seconds=frame_num * 0.1)
                                            
                                            if self.conf_df is not None:
                                                # 首先尝试通过时间直接匹配找到 confidence
                                                try:
                                                    # 找到与 frame_time 完全匹配的时间点
                                                    matching_rows = self.conf_df[self.conf_df['timestamp'] == frame_time]
                                                    if not matching_rows.empty:
                                                        # 如果找到完全匹配的时间点，使用对应的置信度
                                                        confidence = matching_rows.iloc[0]['confidence']
                                                    else:
                                                        # 如果找不到完全匹配的时间点，使用最接近的匹配
                                                        time_diffs = (self.conf_df['timestamp'] - frame_time).abs()
                                                        closest_idx = time_diffs.idxmin()
                                                        confidence = self.conf_df.loc[closest_idx, 'confidence']
                                                except (KeyError, IndexError):
                                                    # 如果取不到使用默认值
                                                    confidence = 0.5
                                                conf_sequence.append(confidence)
                                            else:
                                                # 如果没有提供置信度文件，使用默认值 0.5
                                                conf_sequence.append(0.5)
                                    except Exception as e:
                                        # 如果解析失败，使用默认值 0.5
                                        conf_sequence = [0.5] * sequence_length
                                else:
                                    # 如果文件名格式不正确，使用默认值 0.5
                                    conf_sequence = [0.5] * sequence_length
                                
                                self.samples.append((face_seq['files'], pose_seq['files'], conf_sequence))
                                self.targets.append(face_seq['label'])
                                matched_count += 1
                                break
        
        print(f"成功匹配 {matched_count} 对序列样本")
    
    def __getitem__(self, index):
        face_img_names, pose_img_names, conf_sequence = self.samples[index]
        target = self.targets[index]
        
        from PIL import Image
        face_imgs = []
        pose_imgs = []
        
        # 加载序列中的所有图像
        for face_name, pose_name in zip(face_img_names, pose_img_names):
            face_img = Image.open(os.path.join(face_data_dir, face_name)).convert('RGB')
            pose_img = Image.open(os.path.join(pose_data_dir, pose_name)).convert('RGB')
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
full_dataset = FusionDataset(face_data_dir, pose_data_dir, csv_dir, CONF_CSV_PATH, sequence_length=sequence_length)

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

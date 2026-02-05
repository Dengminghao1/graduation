import glob
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim, autocast
from torch.cuda.amp import GradScaler
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
# 用第二块显卡训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


class MultiSegmentAttentionDataset(Dataset):
    def __init__(self, img_dir, csv_dir, conf_csv_path=None, seq_len=20, transform=None, segment_keys=None):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.label_map = {'低': 0, '稍低': 1, '中性': 2, '稍高': 3, '高': 4}

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
            
        # 2. 加载置信度 CSV 文件（如果提供）
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

        # 2. 解析图像文件并按时间段分组
        segments = defaultdict(list)
        all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        for f in all_files:
            parts = f.split('_')
            # 假设图像文件名: xxxx_frameidx_xxxx_xxxx_开始时间_结束时间.jpg
            # 请根据实际情况确认 parts 的索引
            frame_idx = int(parts[1])
            s_time_str = parts[4]
            e_time_str = parts[5].replace('.jpg', '')

            start_dt = datetime.strptime(s_time_str, "%Y%m%d%H%M%S")
            # curr_dt = start_dt + timedelta(seconds=frame_idx * 0.1)
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

        # 4. 构建序列并从“对应”的 DataFrame 中取标签
        self.valid_sequences = []
        print("开始时序匹配标签...")

        for seg_key, seg_files_list in filtered_segments.items():
            # 检查是否有对应的 CSV 标签
            if seg_key not in self.label_dfs:
                print(f"⚠ 警告: 未找到段 {seg_key} 对应的 CSV 标签，跳过...")
                continue

            current_df = self.label_dfs[seg_key]
            seg_files = sorted(seg_files_list, key=lambda x: x['idx'])

            for i in range(0, len(seg_files) - seq_len,10):
                seq_frames = seg_files[i: i + seq_len]
                end_frame_time = seq_frames[-1]['time']

                # 为序列中的每张图片单独获取置信度
                conf_sequence = []
                valid_sequence = True
                
                # 对于序列中的每张图片
                for frame in seq_frames:
                    frame_time = frame['time']
                    
                    # 从置信度CSV文件中获取置信度
                    if self.conf_df is not None:
                        # 在置信度 DataFrame 中找最接近的时间点
                        time_diffs = (self.conf_df['timestamp'] - frame_time).abs()
                        closest_idx = time_diffs.idxmin()
                        
                        if time_diffs.min() <= timedelta(seconds=1):
                            # 尝试获取 confidence 值，如果不存在则设为默认值 0.5
                            confidence = self.conf_df.loc[closest_idx].get('confidence', 0.5)
                            conf_sequence.append(confidence)
                        else:
                            # 如果找不到对应的时间点，标记序列为无效
                            valid_sequence = False
                            break
                    else:
                        # 如果没有提供置信度文件，使用默认值 0.5
                        confidence = 0.5
                        conf_sequence.append(confidence)
                
                # 如果所有帧都找到了有效的置信度，才保存该序列
                if valid_sequence and conf_sequence:
                    # 使用最后一帧的标签
                    end_frame_time = seq_frames[-1]['time']
                    time_diffs = (current_df['timestamp'] - end_frame_time).abs()
                    closest_idx = time_diffs.idxmin()
                    label_str = current_df.loc[closest_idx, 'attention']
                    
                    self.valid_sequences.append({
                        'files': [x['filename'] for x in seq_frames],
                        'label': self.label_map.get(label_str, 2),
                        'confidence': conf_sequence
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
            frames.append(img)
        # 返回帧序列、标签和置信度序列
        return torch.stack(frames), torch.tensor(data['label'], dtype=torch.long), torch.tensor(data['confidence'], dtype=torch.float32)


# ===========================
# 2. 模型定义 (ResNet50 + LSTM)
# ===========================
class ResNet50QTALSTM(nn.Module):
    def __init__(self, num_classes=5, hidden_size=512):
        super(ResNet50QTALSTM, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)

        self.resnet_out_dim = resnet.fc.in_features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # 新增：特征标准化层，防止 CNN 输出范围波动过大
        self.bn = nn.BatchNorm1d(self.resnet_out_dim)

        # 使用 QTALSTM 替代 LSTM
        self.qlstm = QTALSTM(
            input_size=self.resnet_out_dim,
            hidden_size=hidden_size,
            use_smooth=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, conf):
        b, s, c, h, w = x.shape
        x_flat = x.view(b * s, c, h, w)
        features = self.feature_extractor(x_flat)  # (B*S, 2048, 1, 1)

        # 展平并标准化
        features = features.view(b * s, -1)
        features = self.bn(features)

        # 恢复时序维度
        features = features.view(b, s, -1)

        # 使用 QTALSTM 处理序列，使用外部传递的置信度
        qlstm_out = self.qlstm(features, conf)

        # 取最后一个时间步
        last_timestep_out = qlstm_out[:, -1, :]
        return self.classifier(last_timestep_out)


# ===========================
# 3. 配置与训练脚本
# ===========================
if __name__ == '__main__':
    # --- 配置 ---
    IMG_DIR = r'/home/ccnu/Desktop/dataset/extracted_frames_pic/face_extracted_frames_all'  # <-- 修改这里
    CSV_DIR = r'/home/ccnu/Desktop/dataset/eeg_csv'  # <-- 修改这里
    # 添加置信度CSV文件目录参数
    CONF_CSV_PATH = r'/home/ccnu/Desktop/dataset/confidence.csv'  # <-- 修改这里
    IMG_DIR = r'D:\dataset\frame_picture\face_extracted_frames_101'  # <-- 修改这里
    CSV_DIR = r'D:\dataset\eeg_csv'  # <-- 修改这里
    # 添加置信度CSV文件目录参数
    CONF_CSV_PATH = r"D:\dataset\Dataset_align_face_pose_eeg_feature.csv" # <-- 修改这里
    # IMG_DIR = r'/home/ccnu/Desktop/dataset/frames_face_all'  # <-- 修改这里
    # CSV_DIR = r'/home/ccnu/Desktop/dataset/eeg_csv'  # <-- 修改这里

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 针对 4090 的超参数设置
    BATCH_SIZE = 32  # 增加批量大小以提高泛化能力
    SEQ_LEN = 10  # 输入 1 秒的视频 (10fps * 3s)
    NUM_EPOCHS = 50  # 增加训练轮数
    LEARNING_RATE = 1e-5  # 更小的初始学习率
    NUM_CLASSES = 5

    # 图像预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 1. 数据集切分与加载 ---
    # 假设你的 MultiSegmentAttentionDataset 类已经在上方定义好
    print("正在初始化数据集...")
    # 1. 首先解析出所有的段 key
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    temp_segments = set()
    for f in all_files:
        parts = f.split('_')
        temp_segments.add((parts[4], parts[5].replace('.jpg', '')))
    all_keys = list(temp_segments)

    # # 2. 按“段”划分训练和验证（确保验证集是全新的视频段）
    # train_keys, val_keys = train_test_split(all_keys, test_size=0.2, random_state=42)
    #
    # # 3. 实例化两个独立的 Dataset
    # train_dataset = MultiSegmentAttentionDataset(IMG_DIR, CSV_DIR, SEQ_LEN, data_transforms['train'], segment_keys=train_keys)
    # val_dataset = MultiSegmentAttentionDataset(IMG_DIR, CSV_DIR, SEQ_LEN, data_transforms['val'], segment_keys=val_keys)
    #
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    full_dataset = MultiSegmentAttentionDataset(IMG_DIR, CSV_DIR, CONF_CSV_PATH, SEQ_LEN, data_transforms['train'])

    # 获取数据集长度并进行随机切分
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 使用random_split进行随机切分
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # --- 2. 模型初始化 ---
    model = ResNet50QTALSTM(num_classes=NUM_CLASSES).to(DEVICE)
    # 与resnet_face.py保持一致
    criterion = nn.CrossEntropyLoss()
    # --- 在初始化优化器后添加 ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)  # 增加权重衰减以增强正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)  # 减少 patience 以更激进地衰减学习率
    scaler = GradScaler()

    # 用于绘图的列表
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    print(f"开始训练... 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    patience_counter = 0
    early_stop_patience = 5  # 减少早停耐心值，防止过拟合
    # --- 3. 训练循环 ---
    for epoch in range(NUM_EPOCHS):
        # --- 1. 训练阶段 (Training Phase) ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # 使用 tqdm 包装 train_loader
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        for inputs, labels, conf in train_bar:
            inputs, labels, conf = inputs.to(DEVICE), labels.to(DEVICE), conf.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(inputs, conf)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # 添加梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # 统计数据
            current_batch_size = inputs.size(0)
            train_loss += loss.item() * current_batch_size
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 动态更新进度条右侧的显示信息 (当前 batch 的 loss)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = train_correct / train_total

        # --- 2. 验证阶段 (Validation Phase) ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        # 验证集也建议加上进度条，尤其当验证集较大时
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]")

        with torch.no_grad():
            for inputs, labels, conf in val_bar:
                inputs, labels, conf = inputs.to(DEVICE), labels.to(DEVICE), conf.to(DEVICE)

                with autocast(device_type='cuda'):
                    outputs = model(inputs, conf)
                    v_loss = criterion(outputs, labels)

                val_loss += v_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix(val_loss=f"{v_loss.item():.4f}")

        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_correct / val_total
        scheduler.step(avg_val_loss)  # 自动调整学习率
        # 在训练循环中，在 scheduler.step() 后添加
        if epoch % 1 == 0:  # 每隔一定epoch输出一次
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Learning Rate: {current_lr:.2e}")
        # --- 3. 结果记录与保存 ---
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        # 打印最终汇总结果
        print(f"\nSummary - Epoch [{epoch + 1}/{NUM_EPOCHS}]: "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0  # 重置计数器

            # 清除旧的 best 模型（只删除准确率低于当前最佳的）
            for old_file in glob.glob("best_model_acc_face_qlstm_*.pth"):
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
            save_path = f'best_model_acc_face_qlstm_{acc_suffix}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"🌟 发现更优模型: {save_path}")
        else:
            patience_counter += 1
            print(f"⚠ 验证集表现未提升，早停计数器: {patience_counter}/{early_stop_patience}")

            # 触发早停
            if patience_counter >= early_stop_patience:
                print("🛑 [Early Stopping] 验证集表现长期停滞，提前结束训练。")
                break

    # --- 4. 绘制结果图像 ---
    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results_qlstm.png')
    plt.show()

    print(f"训练结束! 最佳验证集准确率: {best_val_acc:.4f}")
import glob

from matplotlib import pyplot as plt
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


class MultiSegmentAttentionDataset(Dataset):
    def __init__(self, img_dir, csv_path, seq_len=20, transform=None):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform

        # 1. 加载标签
        self.label_df = pd.read_csv(csv_path)
        self.label_df['timestamp'] = pd.to_datetime(self.label_df['timestamp'])
        self.label_map = {'低': 0,
                          '稍低': 1,
                          '中性': 2,
                          '稍高': 3,
                          '高': 4}

        # 2. 解析文件并按“段”分组
        # key: (start_time_str, end_time_str), value: list of file_info
        segments = defaultdict(list)
        all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

        for f in all_files:
            parts = f.split('_')
            # 帧序号: parts[1], 开始时间: parts[4], 结束时间: parts[5]
            frame_idx = int(parts[1])
            s_time_str, e_time_str = parts[4], parts[5].replace('.jpg', '')

            start_dt = datetime.strptime(s_time_str, "%Y%m%d%H%M%S")
            curr_dt = start_dt + timedelta(seconds=frame_idx * 0.1)  # 10 FPS

            segments[(s_time_str, e_time_str)].append({
                'filename': f,
                'time': curr_dt,
                'idx': frame_idx
            })

        # 3. 在每个段内构建连续序列
        self.valid_sequences = []
        all_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

        for f in all_files:
            parts = f.split('_')
            frame_idx = int(parts[1])
            s_time_str, e_time_str = parts[4], parts[5].replace('.jpg', '')

            start_dt = datetime.strptime(s_time_str, "%Y%m%d%H%M%S")
            curr_dt = start_dt + timedelta(seconds=frame_idx * 0.1)

            segments[(s_time_str, e_time_str)].append({
                'filename': f,
                'time': curr_dt,
                'idx': frame_idx
            })

        # 3. 每段内：严格按“1 秒 = 10 帧”构建样本
        self.valid_sequences = []
        print("正在按秒匹配标签并构建序列...")

        for seg_key in segments:
            # 确保段内按帧序号排序
            seg_files = sorted(segments[seg_key], key=lambda x: x['idx'])

            # stride=10 意味着每一秒提取一个序列
            # 如果想让数据更丰富，可以减小 stride；如果想减少冗余，stride 应等于 10
            for i in range(0, len(seg_files) - seq_len, 10):
                seq_frames = seg_files[i: i + seq_len]
                end_frame_time = seq_frames[-1]['time']

                # --- 优化匹配逻辑：寻找 1 秒内最准的那一刻 ---
                time_diffs = (self.label_df['timestamp'] - end_frame_time).abs()
                closest_idx = time_diffs.idxmin()
                min_diff = time_diffs.min()

                if min_diff <= timedelta(seconds=1):
                    label_str = self.label_df.loc[closest_idx, 'attention']
                    self.valid_sequences.append({
                        'files': [x['filename'] for x in seq_frames],
                        'label': self.label_map.get(label_str, 2)
                    })
        print(f"成功创建序列总数: {len(self.valid_sequences)}")

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
        return torch.stack(frames), torch.tensor(data['label'], dtype=torch.long)


# ===========================
# 2. 模型定义 (ResNet50 + LSTM)
# ===========================
class ResNet50LSTM(nn.Module):
    def __init__(self, num_classes=5, hidden_size=512, num_lstm_layers=2):
        super(ResNet50LSTM, self).__init__()

        # 加载预训练的 ResNet50
        # 使用新的 weights API
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)

        # 重要：ResNet50 在全连接层之前的输出维度是 2048
        self.resnet_out_dim = resnet.fc.in_features  # 2048

        # 去掉 ResNet 最后的全连接分类层
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # 定义 LSTM
        self.lstm = nn.LSTM(
            input_size=self.resnet_out_dim,  # 输入维度必须是 2048
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3  # 防止过拟合
        )

        # 定义最终的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入 x 形状: (Batch_Size, Seq_Len, C, H, W)
        b, s, c, h, w = x.shape

        # 1. CNN 特征提取
        # 将 Batch 和 Seq 维度合并，以便并行处理所有图片
        x_flat = x.view(b * s, c, h, w)

        # 注意：这里没有使用 torch.no_grad()，因为 4090 支持全量微调
        features = self.feature_extractor(x_flat)
        # features 形状: (B*S, 2048, 1, 1)

        # 展平并恢复时序维度
        features = features.view(b, s, -1)  # 形状: (B, S, 2048)

        # 2. LSTM 时序建模
        self.lstm.flatten_parameters()  # 优化显存
        lstm_out, _ = self.lstm(features)
        # lstm_out 形状: (B, S, hidden_size)

        # 3. 分类
        # 我们只取最后一个时间步的输出作为整个序列的预测结果
        last_timestep_out = lstm_out[:, -1, :]
        logits = self.classifier(last_timestep_out)

        return logits


# ===========================
# 3. 配置与训练脚本
# ===========================
if __name__ == '__main__':
    # --- 配置 ---
    # r"/home/ccnu/Desktop/2021214387_周婉婷/total/classified_frames"
    IMG_DIR = r'E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\extracted_frames'  # <-- 修改这里
    CSV_PATH = r'D:\GraduationProject\demo1\output\2021214387_周婉婷.csv'  # <-- 修改这里

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 针对 4090 的超参数设置
    BATCH_SIZE = 24  # 24G 显存可以尝试 24 或 32
    SEQ_LEN = 10  # 输入 1 秒的视频 (10fps * 3s)
    NUM_EPOCHS = 15
    LEARNING_RATE = 3e-5  # 微调时学习率要小
    NUM_CLASSES = 5

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 1. 数据集切分与加载 ---
    # 假设你的 MultiSegmentAttentionDataset 类已经在上方定义好
    print("正在初始化数据集...")
    full_dataset = MultiSegmentAttentionDataset(img_dir=IMG_DIR, csv_path=CSV_PATH, seq_len=SEQ_LEN,
                                                transform=transform)

    # 按照 8:2 切分训练集和验证集
    # 注意：对于视频，更好的方式是按视频文件切分，这里先使用随机切分索引
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # --- 2. 模型初始化 ---
    model = ResNet50LSTM(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()

    # 用于绘图的列表
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    print(f"开始训练... 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # --- 3. 训练循环 ---
    for epoch in range(NUM_EPOCHS):
        # --- 1. 训练阶段 (Training Phase) ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # 使用 tqdm 包装 train_loader
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train")

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
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
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Val")

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    v_loss = criterion(outputs, labels)

                val_loss += v_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix(val_loss=f"{v_loss.item():.4f}")

        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_correct / val_total

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
            for old_file in glob.glob("best_model_acc_*.pth"):
                os.remove(old_file)

            acc_suffix = int(best_val_acc * 10000)
            save_path = f'best_model_acc_{acc_suffix}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"🌟 发现更优模型: {save_path}")

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
    plt.savefig('training_results.png')
    plt.show()

    print(f"训练结束! 最佳验证集准确率: {best_val_acc:.4f}")
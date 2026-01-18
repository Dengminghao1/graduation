import glob
from torch import nn, optim, autocast
from torch.cuda.amp import GradScaler
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from datetime import datetime, timedelta
from collections import defaultdict


class MultiSegmentAttentionDataset(Dataset):
    def __init__(self, img_dir, csv_path, seq_len=20, transform=None):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform

        # 1. 加载标签
        self.label_df = pd.read_csv(csv_path)
        self.label_df['timestamp'] = pd.to_datetime(self.label_df['timestamp'])
        self.label_map = {'低': 1,
                          '稍低': 2,
                          '中性': 3,
                          '稍高': 4,
                          '高': 5}

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
        for seg_key in segments:
            # 确保段内按帧序号排序
            seg_files = sorted(segments[seg_key], key=lambda x: x['idx'])

            # 在当前段内滑动窗口
            for i in range(len(seg_files) - seq_len):
                seq_frames = seg_files[i: i + seq_len]
                end_frame_time = seq_frames[-1]['time']

                # 匹配最接近的标签（精确到秒）
                # 寻找标签时间与帧时间误差在1秒以内的记录
                matched = self.label_df[
                    (self.label_df['timestamp'] >= end_frame_time - timedelta(seconds=1)) &
                    (self.label_df['timestamp'] <= end_frame_time + timedelta(seconds=1))
                    ]

                if not matched.empty:
                    label_str = matched.iloc[-1]['attention']
                    self.valid_sequences.append({
                        'files': [x['filename'] for x in seq_frames],
                        'label': self.label_map.get(label_str, 3)  # 默认“一般”
                    })
        print(f"Total sequences created: {len(self.valid_sequences)}")

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
    IMG_DIR = r'E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\extracted_frames'  # <-- 修改这里
    CSV_PATH = r'D:\GraduationProject\demo1\output\2021214387_周婉婷.csv'  # <-- 修改这里

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 针对 4090 的超参数设置
    BATCH_SIZE = 24  # 24G 显存可以尝试 24 或 32
    SEQ_LEN = 30  # 输入 3 秒的视频 (10fps * 3s)
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

    # --- 实例化数据和模型 ---
    dataset = MultiSegmentAttentionDataset(img_dir=IMG_DIR, csv_path=CSV_PATH, seq_len=SEQ_LEN, transform=transform)

    # num_workers=8 利用多核CPU加速数据读取，pin_memory=True 加速数据传入GPU
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = ResNet50LSTM(num_classes=NUM_CLASSES).to(DEVICE)

    # 损失函数和优化器
    # 如果你的数据类别严重不平衡（例如“高”特别多），考虑给 CrossEntropyLoss 添加 weight 参数
    criterion = nn.CrossEntropyLoss()
    # 使用 AdamW 优化器，对微调效果更好
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 混合精度训练 GradScaler
    scaler = GradScaler()

    # --- 训练准备 ---
    best_acc = 0.0  # 初始化最高准确率为0
    best_model_name = ""

    print("开始训练...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计信息
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}, Step {i + 1}] Loss: {loss.item():.4f}")

        # 计算该 Epoch 的平均指标
        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct_preds / total_preds
        print(f"--- Epoch {epoch + 1} Finished. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} ---")

        # --- 核心修改：仅保存最好的模型 ---
        if epoch_acc > best_acc:
            # 1. 更新最高准确率记录
            best_acc = epoch_acc

            # 2. 删除之前保存过的 best_model 文件（防止硬盘堆积）
            # 搜索目录下所有以 'best_model_acc_' 开头的文件并删除
            for old_file in glob.glob("best_model_acc_*.pth"):
                try:
                    os.remove(old_file)
                except:
                    pass

                    # 3. 构造新的文件名并保存
            # 例如：best_model_acc_9542.pth
            save_path = f'best_model_acc_{int(best_acc*10000)}.pth'
            torch.save(model.state_dict(), save_path)

            print(f"🌟 检测到更好的模型！准确率提高到: {best_acc:.4f}，已保存为 {save_path}")
        else:
            print(f"ℹ️ 本轮准确率 ({epoch_acc:.4f}) 未超过历史最好成绩 ({best_acc:.4f})，不保存。")

    print(f"训练结束! 最好的模型准确率为: {best_acc:.4f}")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取标准化后的数据
data_path = 'd:\\Pycharm_Projects\\demo1_trae\\Dataset_align_face_pose_eeg_feature_standardized.csv'

# 读取标签（使用分块读取以节省内存）
labels = []
for chunk in pd.read_csv(data_path, chunksize=10000, usecols=['attention']):
    labels.extend(chunk['attention'].tolist())

if not labels:
    # 如果没有attention列，使用一个示例标签（实际应用中需要替换）
    # 先计算文件总行数
    total_rows = 0
    for chunk in pd.read_csv(data_path, chunksize=10000):
        total_rows += len(chunk)
    labels = np.random.randint(0, 2, total_rows)

# 数据预处理
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))

# 划分训练集和测试集
train_indices, test_indices = train_test_split(
    np.arange(len(labels_encoded)), test_size=0.2, random_state=42
)

# 提取肢体特征列名（从x0到y18）
pose_cols = []
# 读取第一行来获取列名
first_chunk = next(pd.read_csv(data_path, chunksize=1))
columns = first_chunk.columns.tolist()

# 直接指定从x0到y18的列
if 'x0' in columns and 'y18' in columns:
    start_idx = columns.index('x0')
    end_idx = columns.index('y18')
    pose_cols = columns[start_idx:end_idx+1]
    print(f"成功提取pose特征列: {len(pose_cols)}列")
    print(f"特征范围: {pose_cols[0]} 到 {pose_cols[-1]}")
else:
    # 如果找不到x0或y18，使用备用方法
    for col in columns:
        if col.startswith('x') or col.startswith('y'):
            try:
                if col == 'x0' or (pose_cols and col.startswith(pose_cols[-1][0]) and int(col[1:]) == int(pose_cols[-1][1:]) + 1):
                    pose_cols.append(col)
                    if col == 'y18':
                        break
            except:
                pass
    print(f"备用方法提取pose特征列: {len(pose_cols)}列")

# 构建全连接模型
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化模型、损失函数和优化器
# 先获取输入特征数量
first_chunk = next(pd.read_csv(data_path, chunksize=1, usecols=pose_cols))
input_size = first_chunk.shape[1]
model = Classifier(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
epochs = 50
batch_size = 32
validation_split = 0.1

# 划分验证集
val_size = int(len(train_indices) * validation_split)
train_indices, val_indices = train_test_split(
    train_indices, test_size=val_size, random_state=42
)

# 训练历史
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# 训练模型
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 批处理
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size != 0 else 0)
    with tqdm(total=total_batches, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            
            # 从文件中加载批次数据
            batch_data = []
            batch_targets = []
            
            for chunk in pd.read_csv(data_path, chunksize=10000, usecols=pose_cols):
                start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
                end_idx = start_idx + len(chunk)
                chunk_indices = [idx for idx in batch_indices if start_idx <= idx < end_idx]
                if chunk_indices:
                    relative_indices = [idx - start_idx for idx in chunk_indices]
                    batch_data.extend(chunk.iloc[relative_indices].values.tolist())
                    batch_targets.extend([labels_encoded[idx] for idx in chunk_indices])
                if len(batch_data) >= len(batch_indices):
                    break
            
            # 转换为张量
            inputs = torch.tensor(batch_data, dtype=torch.float32).to(device)
            targets = torch.tensor(batch_targets, dtype=torch.long).to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # 更新进度条
            current_loss = running_loss / (i // batch_size + 1)
            current_acc = correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
            pbar.update(1)
    
    # 计算训练损失和准确率
    train_loss = running_loss / (len(train_indices) // batch_size + 1)
    train_acc = correct / total
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    
    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for i in range(0, len(val_indices), batch_size):
            batch_indices = val_indices[i:i+batch_size]
            
            # 从文件中加载批次数据
            batch_data = []
            batch_targets = []
            
            for chunk in pd.read_csv(data_path, chunksize=10000, usecols=pose_cols):
                start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
                end_idx = start_idx + len(chunk)
                chunk_indices = [idx for idx in batch_indices if start_idx <= idx < end_idx]
                if chunk_indices:
                    relative_indices = [idx - start_idx for idx in chunk_indices]
                    batch_data.extend(chunk.iloc[relative_indices].values.tolist())
                    batch_targets.extend([labels_encoded[idx] for idx in chunk_indices])
                if len(batch_data) >= len(batch_indices):
                    break
            
            # 转换为张量
            inputs = torch.tensor(batch_data, dtype=torch.float32).to(device)
            targets = torch.tensor(batch_targets, dtype=torch.long).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
    
    val_loss = val_loss / (len(val_indices) // batch_size + 1)
    val_acc = val_correct / val_total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# 评估模型
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
y_pred = []
y_true = []

with torch.no_grad():
    for i in range(0, len(test_indices), batch_size):
        batch_indices = test_indices[i:i+batch_size]
        
        # 从文件中加载批次数据
        batch_data = []
        batch_targets = []
        
        for chunk in pd.read_csv(data_path, chunksize=10000, usecols=pose_cols):
            start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
            end_idx = start_idx + len(chunk)
            chunk_indices = [idx for idx in batch_indices if start_idx <= idx < end_idx]
            if chunk_indices:
                relative_indices = [idx - start_idx for idx in chunk_indices]
                batch_data.extend(chunk.iloc[relative_indices].values.tolist())
                batch_targets.extend([labels_encoded[idx] for idx in chunk_indices])
            if len(batch_data) >= len(batch_indices):
                break
        
        # 转换为张量
        inputs = torch.tensor(batch_data, dtype=torch.float32).to(device)
        targets = torch.tensor(batch_targets, dtype=torch.long).to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += targets.size(0)
        test_correct += (predicted == targets).sum().item()
        
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(targets.cpu().numpy())

accuracy = test_correct / test_total
print(f'Test Loss: {test_loss / (len(test_indices) // batch_size + 1):.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# 计算评估指标
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# 绘制训练损失和准确率
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('pose_training_metrics.png')
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Pose Features')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('pose_confusion_matrix.png')
plt.show()

# T-SNE可视化
print('Performing t-SNE visualization...')
# 取前1000个样本进行可视化
n_samples = min(1000, len(test_indices))
test_indices_subset = test_indices[:n_samples]

# 加载用于t-SNE的样本
X_tsne = []
y_tsne = []

for chunk in pd.read_csv(data_path, chunksize=10000, usecols=pose_cols):
    start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
    end_idx = start_idx + len(chunk)
    chunk_indices = [idx for idx in test_indices_subset if start_idx <= idx < end_idx]
    if chunk_indices:
        relative_indices = [idx - start_idx for idx in chunk_indices]
        X_tsne.extend(chunk.iloc[relative_indices].values.tolist())
        y_tsne.extend([labels_encoded[idx] for idx in chunk_indices])
    if len(X_tsne) >= n_samples:
        break

# 应用t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne_embedded = tsne.fit_transform(X_tsne)

# 绘制t-SNE结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne_embedded[:, 0], X_tsne_embedded[:, 1], c=y_tsne, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE Visualization - Pose Features')
plt.savefig('pose_tsne_visualization.png')
plt.show()

print('Training completed successfully!')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
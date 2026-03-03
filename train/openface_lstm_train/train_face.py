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

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取标准化后的数据
data_path = 'd:\\Pycharm_Projects\\demo1_trae\\Dataset_align_face_pose_eeg_feature_standardized.csv'

# 首先读取列名，确定特征范围
with open(data_path, 'r') as f:
    header = f.readline().strip().split(',')

# 确定面部特征的列索引
face_start = header.index('gaze_0_x')
face_end = header.index('p_33') + 1
face_cols = header[face_start:face_end]

# 逐块读取数据，计算标签
print('Reading labels...')
labels = []
chunk_size = 10000
for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=['attention']):
    if 'attention' in chunk.columns:
        labels.extend(chunk['attention'].tolist())
    else:
        # 如果没有attention列，使用示例标签
        labels.extend(np.random.randint(0, 2, len(chunk)).tolist())

# 数据预处理
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))

# 划分训练集和测试集的索引
print('Splitting data...')
indices = np.arange(len(labels))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

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
input_size = len(face_cols)
model = Classifier(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
epochs = 50
batch_size = 1024
validation_split = 0.1

# 划分验证集
train_size = len(train_indices)
val_size = int(train_size * validation_split)
train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=42)

# 训练历史
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# 训练模型
print('Training model...')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 打乱训练索引
    np.random.shuffle(train_indices)
    
    # 批处理训练数据
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size != 0 else 0)
    with tqdm(total=total_batches, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:min(i + batch_size, len(train_indices))]
            
            # 读取批次数据
            # 由于内存限制，我们使用chunksize来读取数据
            batch_data = []
            for chunk in pd.read_csv(data_path, chunksize=10000, usecols=face_cols):
                # 计算当前chunk的索引范围
                start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
                end_idx = start_idx + len(chunk)
                
                # 检查是否有批次索引在当前chunk中
                chunk_indices = [idx for idx in batch_indices if start_idx <= idx < end_idx]
                if chunk_indices:
                    # 计算在chunk中的相对索引
                    relative_indices = [idx - start_idx for idx in chunk_indices]
                    batch_data.extend(chunk.iloc[relative_indices].values.tolist())
                
                # 如果已经收集了所有批次数据，停止读取
                if len(batch_data) >= len(batch_indices):
                    break
            
            batch_data = np.array(batch_data)
            batch_labels = labels_encoded[batch_indices]
            
            # 处理NaN值，替换为0
            batch_data = np.nan_to_num(batch_data, nan=0.0)
            
            # 转换为PyTorch张量
            inputs = torch.tensor(batch_data, dtype=torch.float32).to(device)
            targets = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
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
            batch_indices = val_indices[i:min(i + batch_size, len(val_indices))]
            
            # 读取批次数据
            batch_data = []
            for chunk in pd.read_csv(data_path, chunksize=10000, usecols=face_cols):
                start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
                end_idx = start_idx + len(chunk)
                chunk_indices = [idx for idx in batch_indices if start_idx <= idx < end_idx]
                if chunk_indices:
                    relative_indices = [idx - start_idx for idx in chunk_indices]
                    batch_data.extend(chunk.iloc[relative_indices].values.tolist())
                if len(batch_data) >= len(batch_indices):
                    break
            
            batch_data = np.array(batch_data)
            batch_labels = labels_encoded[batch_indices]
            
            # 处理NaN值，替换为0
            batch_data = np.nan_to_num(batch_data, nan=0.0)
            
            # 转换为PyTorch张量
            inputs = torch.tensor(batch_data, dtype=torch.float32).to(device)
            targets = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
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
print('Evaluating model...')
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for i in range(0, len(test_indices), batch_size):
        batch_indices = test_indices[i:min(i + batch_size, len(test_indices))]
        
        # 读取批次数据
        batch_data = []
        for chunk in pd.read_csv(data_path, chunksize=10000, usecols=face_cols):
            start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
            end_idx = start_idx + len(chunk)
            chunk_indices = [idx for idx in batch_indices if start_idx <= idx < end_idx]
            if chunk_indices:
                relative_indices = [idx - start_idx for idx in chunk_indices]
                batch_data.extend(chunk.iloc[relative_indices].values.tolist())
            if len(batch_data) >= len(batch_indices):
                break
        
        batch_data = np.array(batch_data)
        batch_labels = labels_encoded[batch_indices]
        
        # 转换为PyTorch张量
        inputs = torch.tensor(batch_data, dtype=torch.float32).to(device)
        targets = torch.tensor(batch_labels, dtype=torch.long).to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Test Accuracy: {accuracy:.4f}')
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
plt.savefig('face_training_metrics.png')
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Face Features')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('face_confusion_matrix.png')
plt.show()

# T-SNE可视化
print('Performing t-SNE visualization...')
# 取前1000个测试样本进行可视化
n_samples = min(1000, len(test_indices))
sample_indices = test_indices[:n_samples]

# 读取样本数据
sample_data = []
for chunk in pd.read_csv(data_path, chunksize=10000, usecols=face_cols):
    start_idx = chunk.index[0] if hasattr(chunk, 'index') else 0
    end_idx = start_idx + len(chunk)
    chunk_indices = [idx for idx in sample_indices if start_idx <= idx < end_idx]
    if chunk_indices:
        relative_indices = [idx - start_idx for idx in chunk_indices]
        sample_data.extend(chunk.iloc[relative_indices].values.tolist())
    if len(sample_data) >= n_samples:
        break

sample_data = np.array(sample_data)
y_tsne = labels_encoded[sample_indices]

# 处理NaN值，替换为0
sample_data = np.nan_to_num(sample_data, nan=0.0)

# 应用t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne_embedded = tsne.fit_transform(sample_data)

# 绘制t-SNE结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne_embedded[:, 0], X_tsne_embedded[:, 1], c=y_tsne, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE Visualization - Face Features')
plt.savefig('face_tsne_visualization.png')
plt.show()

print('Training completed successfully!')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
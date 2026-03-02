import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import psutil

# 配置参数
face_data_dir = r"/home/ccnu/Desktop/dataset/extracted_frames_pic/face_extracted_frames_all"  # 面部数据
pose_data_dir = r"/home/ccnu/Desktop/dataset/extracted_frames_pic/pose_224_all"  # 肢体数据
csv_dir = r"/home/ccnu/Desktop/dataset/eeg_csv"  # 标签CSV文件目录
output_dir = r"/home/ccnu/Desktop/dataset/preprocessed_data"  # 预处理后数据保存目录

# # 本地测试路径
# face_data_dir = r"D:\dataset\20231229153000_20231229154000_face"  # 面部数据
# pose_data_dir = r"D:\dataset\20231229153000_20231229154000_pose"  # 肢体数据
# csv_dir = r"D:\dataset\eeg_csv"  # 标签CSV文件目录
# output_dir = r"D:\dataset\preprocessed_data"  # 预处理后数据保存目录

sequence_length = 10  # 帧序列长度
batch_size = 20  # 预处理批次大小
num_workers = 4  # 并行处理线程数

# 数据增强和变换
data_transforms = {
    'face_train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
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
}

label_map = {'低': 0, '稍低': 1, '中性': 2, '稍高': 3, '高': 4}

class Preprocessor:
    def __init__(self, face_data_dir, pose_data_dir, csv_dir, output_dir):
        self.face_data_dir = face_data_dir
        self.pose_data_dir = pose_data_dir
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.label_dfs = {}
        self.face_segments = defaultdict(list)
        self.pose_segments = defaultdict(list)
        
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, 'face'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'pose'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
    def load_label_dfs(self):
        """加载标签CSV文件"""
        csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        print(f"正在加载 {len(csv_files)} 个标签文件...")
        
        for cf in csv_files:
            parts = cf.replace('.csv', '').split('_')
            s_str, e_str = parts[-2], parts[-1]
            
            df = pd.read_csv(os.path.join(self.csv_dir, cf))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.label_dfs[(s_str, e_str)] = df
    
    def parse_image_files(self):
        """解析图像文件并按时间段分组"""
        # 解析面部图像
        face_files = [f for f in os.listdir(self.face_data_dir) if f.endswith('.jpg')]
        print(f"正在解析 {len(face_files)} 个面部图像文件...")
        
        for f in face_files:
            parts = f.split('_')
            if len(parts) >= 5:
                frame_idx = int(parts[1])
                s_time_str = parts[-2]
                e_time_str = parts[-1].replace('.jpg', '')
                
                start_dt = datetime.strptime(s_time_str, "%Y%m%d%H%M%S")
                total_milliseconds = frame_idx * 100
                curr_dt = start_dt + timedelta(milliseconds=total_milliseconds)
                
                self.face_segments[(s_time_str, e_time_str)].append({
                    'filename': f,
                    'time': curr_dt,
                    'idx': frame_idx
                })
        
        # 解析肢体图像
        pose_files = [f for f in os.listdir(self.pose_data_dir) if f.endswith('.png')]
        print(f"正在解析 {len(pose_files)} 个肢体图像文件...")
        
        for f in pose_files:
            parts = f.split('_')
            if len(parts) >= 6:
                frame_idx = int(parts[-2])
                s_time_str = parts[-4]
                e_time_str = parts[-3]
                
                start_dt = datetime.strptime(s_time_str, "%Y%m%d%H%M%S")
                total_milliseconds = frame_idx * 100
                curr_dt = start_dt + timedelta(milliseconds=total_milliseconds)
                
                self.pose_segments[(s_time_str, e_time_str)].append({
                    'filename': f,
                    'time': curr_dt,
                    'idx': frame_idx
                })
    
    def match_sequences(self):
        """匹配面部和肢体序列并生成预处理数据"""
        print("开始匹配面部和肢体序列...")
        matched_sequences = []
        
        for seg_key in self.face_segments:
            if seg_key not in self.pose_segments:
                continue
            if seg_key not in self.label_dfs:
                continue
            
            face_files = sorted(self.face_segments[seg_key], key=lambda x: x['idx'])
            pose_files = sorted(self.pose_segments[seg_key], key=lambda x: x['idx'])
            current_df = self.label_dfs[seg_key]
            
            # 生成序列
            for i in range(0, len(face_files) - sequence_length + 1, 10):
                face_seq = face_files[i: i + sequence_length]
                end_frame_time = face_seq[-1]['time']
                
                # 查找匹配的肢体序列
                for j in range(0, len(pose_files) - sequence_length + 1, 10):
                    pose_seq = pose_files[j: j + sequence_length]
                    if abs(face_seq[0]['idx'] - pose_seq[0]['idx']) == 0:
                        # 查找标签
                        time_diffs = (current_df['timestamp'] - end_frame_time).abs()
                        closest_idx = time_diffs.idxmin()
                        label_str = current_df.loc[closest_idx, 'attention']
                        label = label_map.get(label_str, 2)
                        
                        matched_sequences.append({
                            'face_files': [x['filename'] for x in face_seq],
                            'pose_files': [x['filename'] for x in pose_seq],
                            'label': label
                        })
                        break
        
        print(f"成功匹配 {len(matched_sequences)} 对序列样本")
        return matched_sequences
    
    def preprocess_batch(self, batch_sequences, batch_idx):
        """预处理批次数据并保存为.pt文件"""
        face_data = []
        pose_data = []
        labels = []
        
        for seq in batch_sequences:
            # 处理面部序列
            face_frames = []
            for fname in seq['face_files']:
                img = Image.open(os.path.join(self.face_data_dir, fname)).convert('RGB')
                img = data_transforms['face_train'](img)
                face_frames.append(img)
            face_data.append(torch.stack(face_frames))
            
            # 处理肢体序列
            pose_frames = []
            for fname in seq['pose_files']:
                img = Image.open(os.path.join(self.pose_data_dir, fname)).convert('RGB')
                img = data_transforms['pose_train'](img)
                pose_frames.append(img)
            pose_data.append(torch.stack(pose_frames))
            
            labels.append(seq['label'])
        
        # 转换为张量
        face_data = torch.stack(face_data)
        pose_data = torch.stack(pose_data)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # 保存为.pt文件
        torch.save(face_data, os.path.join(self.output_dir, 'face', f'face_batch_{batch_idx}.pt'))
        torch.save(pose_data, os.path.join(self.output_dir, 'pose', f'pose_batch_{batch_idx}.pt'))
        torch.save(labels, os.path.join(self.output_dir, 'labels', f'labels_batch_{batch_idx}.pt'))
        
        # 释放内存
        del face_data, pose_data, labels
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def preprocess(self):
        """执行完整的预处理流程"""
        # 加载标签
        self.load_label_dfs()
        
        # 解析图像文件
        self.parse_image_files()
        
        # 匹配序列
        matched_sequences = self.match_sequences()
        
        # 批量预处理
        print("开始批量预处理数据...")
        total_batches = (len(matched_sequences) + batch_size - 1) // batch_size
        
        for i in tqdm(range(total_batches), desc="预处理批次"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(matched_sequences))
            batch_sequences = matched_sequences[start_idx:end_idx]
            
            # 监控内存使用
            current_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)
            if current_mem > 8.0:
                print(f"⚠ 警告: 内存使用过高 ({current_mem:.2f} GB)")
            
            self.preprocess_batch(batch_sequences, i)
        
        # 保存元数据
        metadata = {
            'total_sequences': len(matched_sequences),
            'batch_size': batch_size,
            'total_batches': total_batches
        }
        torch.save(metadata, os.path.join(self.output_dir, 'metadata.pt'))
        print("预处理完成！")

if __name__ == "__main__":
    preprocessor = Preprocessor(face_data_dir, pose_data_dir, csv_dir, output_dir)
    preprocessor.preprocess()

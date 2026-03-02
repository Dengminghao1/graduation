import pandas as pd
import numpy as np
import os

# 读取CSV文件的列名，获取特征范围
file_path = 'd:\\dataset\\Dataset_align_face_pose_eeg_feature.csv'

# 首先读取列名
with open(file_path, 'r') as f:
    header = f.readline().strip().split(',')

# 确定面部特征和肢体特征的列索引
face_start = header.index('gaze_0_x')
face_end = header.index('p_33') + 1
body_start = header.index('x0')
body_end = header.index('y18') + 1

# 计算特征列
face_cols = header[face_start:face_end]
body_cols = header[body_start:body_end]

print(f"面部特征列数: {len(face_cols)}")
print(f"肢体特征列数: {len(body_cols)}")

# 分块读取数据并计算均值和标准差
chunk_size = 10000
face_means = np.zeros(len(face_cols))
face_stds = np.zeros(len(face_cols))
body_means = np.zeros(len(body_cols))
body_stds = np.zeros(len(body_cols))
total_rows = 0

print("计算特征均值和标准差...")
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunk_rows = len(chunk)
    total_rows += chunk_rows
    
    # 计算面部特征的均值和标准差
    face_chunk = chunk[face_cols]
    face_means += face_chunk.mean().values * chunk_rows
    face_stds += face_chunk.std().values * chunk_rows
    
    # 计算肢体特征的均值和标准差
    body_chunk = chunk[body_cols]
    body_means += body_chunk.mean().values * chunk_rows
    body_stds += body_chunk.std().values * chunk_rows

# 计算总体均值和标准差
face_means /= total_rows
face_stds /= total_rows
body_means /= total_rows
body_stds /= total_rows

print(f"总数据行数: {total_rows}")

# 再次分块读取数据并进行标准化处理
output_file = 'd:\\dataset\\Dataset_align_face_pose_eeg_feature_standardized.csv'
print(f"进行特征标准化并保存结果到: {output_file}")

# 写入表头
with open(output_file, 'w') as f:
    f.write(','.join(header) + '\n')

# 分块处理并写入结果
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    # 标准化面部特征
    face_chunk = chunk[face_cols]
    face_standardized = (face_chunk - face_means) / face_stds
    
    # 标准化肢体特征
    body_chunk = chunk[body_cols]
    body_standardized = (body_chunk - body_means) / body_stds
    
    # 替换原始数据中的特征值
    chunk[face_cols] = face_standardized
    chunk[body_cols] = body_standardized
    
    # 写入文件，保留6位小数
    if i == 0:
        chunk.to_csv(output_file, mode='a', index=False, header=False, float_format='%.6f')
    else:
        chunk.to_csv(output_file, mode='a', index=False, header=False, float_format='%.6f')
    
    if (i + 1) % 10 == 0:
        print(f"已处理 {i * chunk_size + len(chunk)} 行数据")

print(f"特征标准化完成，结果保存到: {os.path.abspath(output_file)}")
print(f"标准化前面部特征均值: {face_means.mean():.6f}")
print(f"标准化前面部特征标准差: {face_stds.mean():.6f}")
print(f"标准化前肢体特征均值: {body_means.mean():.6f}")
print(f"标准化前肢体特征标准差: {body_stds.mean():.6f}")
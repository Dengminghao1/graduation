import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta


def get_start_time_from_filename(filename):
    """
    从文件名中提取起始时间戳
    例如: per_second_192.168.0.101_01_20231229150516_20231229151709.csv
    提取 20231229150516
    """
    match = re.search(r'_(\d{14})_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return None


def align_face_with_eeg(eeg_file, face_file, output_file):
    """
    将脑电数据与面部识别数据按照时间对齐
    """
    print(f"\n正在处理文件对:")
    print(f"  脑电: {os.path.basename(eeg_file)}")
    print(f"  面部: {os.path.basename(face_file)}")

    # 1. 读取脑电数据
    eeg_data = pd.read_csv(eeg_file)
    if eeg_data.columns[0].startswith('﻿'):  # 处理 BOM
        eeg_data = eeg_data.rename(columns={eeg_data.columns[0]: eeg_data.columns[0].replace('﻿', '')})
    eeg_data.columns = ['timestamp', 'attention']
    eeg_data['timestamp'] = pd.to_datetime(eeg_data['timestamp'])

    # 2. 读取面部识别数据
    face_data = pd.read_csv(face_file)
    face_data.columns = [col.strip() for col in face_data.columns]

    # 3. 动态生成 actual_time
    # 获取文件名中的起始时间
    video_start_dt = get_start_time_from_filename(os.path.basename(face_file))

    if video_start_dt and 'timestamp' in face_data.columns:
        # 核心逻辑：起始时间 + timestamp(偏移秒数)
        face_data['actual_time'] = face_data['timestamp'].apply(lambda x: video_start_dt + timedelta(seconds=float(x)))
        print(f"  已生成 actual_time，范围: {face_data['actual_time'].min()} 到 {face_data['actual_time'].max()}")
    else:
        print(f"  [错误] 无法从文件名提取时间或 CSV 中缺少 timestamp 列")
        return

    # 4. 排序 (merge_asof 要求有序)
    eeg_data = eeg_data.sort_values('timestamp')
    face_data = face_data.sort_values('actual_time')

    # 5. 执行最近邻对齐
    # tolerance='1000ms' 表示允许 1 秒误差
    aligned_df = pd.merge_asof(
        face_data,
        eeg_data,
        left_on='actual_time',
        right_on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1000ms')
    )
    # 6. 重命名时间戳列
    aligned_df = aligned_df.rename(columns={
        'timestamp_x': 'timestamp_face',
        'timestamp_y': 'timestamp_eeg'
    })

    # 6. 过滤掉没有脑电标签的行并保存
    aligned_df_filtered = aligned_df[aligned_df['attention'].notna()]
    aligned_df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"  对齐完成！匹配成功: {len(aligned_df_filtered)} / {len(face_data)} 帧")


def batch_process_face_eeg(eeg_dir, face_dir, output_dir):
    """批量处理逻辑"""
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r'(\d{14}_\d{14})')

    eeg_files = {pattern.search(f).group(1): os.path.join(eeg_dir, f)
                 for f in os.listdir(eeg_dir) if pattern.search(f)}

    face_files = {pattern.search(f).group(1): os.path.join(face_dir, f)
                  for f in os.listdir(face_dir) if pattern.search(f)}

    for t_key, face_path in face_files.items():
        if t_key in eeg_files:
            out_name = f"aligned_face_eeg_{t_key}.csv"
            align_face_with_eeg(eeg_files[t_key], face_path, os.path.join(output_dir, out_name))


if __name__ == "__main__":
    # 配置路径
    eeg_directory = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs_csv"
    face_directory = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\face_feature"
    output_directory = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\align_face_eeg"

    batch_process_face_eeg(eeg_directory, face_directory, output_directory)
    # eeg_file = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs_csv\2021214387_周婉婷_20231229150516_20231229151709.csv"
    # faces_file = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\face_feature\192.168.0.101_01_20231229150516_20231229151709.csv"
    # output_file = r"D:\GraduationProject\demo1\output\align_eeg_face.csv"
    # align_face_with_eeg(eeg_file, faces_file, output_file)
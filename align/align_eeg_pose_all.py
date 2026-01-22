import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
from functools import partial


def extract_time_from_kp_row(file_name_cell, fps=10):
    """
    解析关键点 CSV 中 file_name 列的时间
    fps: 视频帧率，用于计算时间偏移
    """
    parts = file_name_cell.split('_')
    try:
        # parts[2] = "20231229150516" (起始时间戳)
        start_time_str = parts[2]
        # parts[4] = "000000000005" (帧序号)
        frame_idx = int(parts[4])

        start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")

        # 计算偏移：帧序号 * (1 / 帧率)
        actual_time = start_time + timedelta(seconds=frame_idx * (1.0 / fps))
        return actual_time
    except (IndexError, ValueError):
        return None


def align_eeg_with_keypoints(eeg_file, keypoints_file, output_file, fps=10):
    """将脑电数据与关键点 CSV 对齐"""

    print(f"\n正在处理文件对 (FPS: {fps}):")
    print(f"  关键点: {os.path.basename(keypoints_file)}")
    print(f"  脑电:   {os.path.basename(eeg_file)}")

    # 1. 读取数据
    eeg_data = pd.read_csv(eeg_file)
    if eeg_data.columns[0].startswith('﻿'):
        eeg_data = eeg_data.rename(columns={eeg_data.columns[0]: eeg_data.columns[0].replace('﻿', '')})
    eeg_data.columns = ['timestamp', 'attention']
    eeg_data['timestamp'] = pd.to_datetime(eeg_data['timestamp'])

    kp_data = pd.read_csv(keypoints_file)

    # 2. 计算每一帧的精确时间 (传递 fps 参数)
    kp_data['actual_time'] = kp_data['file_name'].apply(lambda x: extract_time_from_kp_row(x, fps=fps))

    # 移除无法解析时间的行
    kp_data = kp_data.dropna(subset=['actual_time'])

    # 3. 排序（merge_asof 必须要求有序）
    eeg_data = eeg_data.sort_values('timestamp')
    kp_data = kp_data.sort_values('actual_time')

    # 4. 毫秒级对齐
    aligned_df = pd.merge_asof(
        kp_data,
        eeg_data,
        left_on='actual_time',
        right_on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1000ms')
    )

    # 5. 重命名时间戳列
    aligned_df = aligned_df.rename(columns={
        'actual_time': 'timestamp_pose',
        'timestamp': 'timestamp_eeg'
    })

    # 6. 过滤并保存
    aligned_df_filtered = aligned_df[aligned_df['attention'].notna()]
    aligned_df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')

    valid_matches = len(aligned_df_filtered)
    print(f"  对齐完成！有效匹配行数: {valid_matches} / 总行数: {len(aligned_df)}")


def batch_process(eeg_dir, kp_dir, output_dir, fps=10):
    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r'(\d{14}_\d{14})')

    eeg_files = {pattern.search(f).group(1): os.path.join(eeg_dir, f)
                 for f in os.listdir(eeg_dir) if pattern.search(f)}

    kp_files = {pattern.search(f).group(1): os.path.join(kp_dir, f)
                for f in os.listdir(kp_dir) if pattern.search(f)}

    for t_key, kp_path in kp_files.items():
        if t_key in eeg_files:
            out_name = f"aligned_eeg_pose_{t_key}.csv"
            # 向下单向传递 fps
            align_eeg_with_keypoints(eeg_files[t_key], kp_path, os.path.join(output_dir, out_name), fps=fps)
        else:
            print(f"跳过: 找不到与 {t_key} 匹配的脑电文件")


if __name__ == "__main__":



    # eeg_file = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs_csv\2021214387_周婉婷_20231229150516_20231229151709.csv"

    # keypoints_file = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\pose_feature_merged\192.168.0.101_01_20231229150516_20231229151709.csv"

    # output_file = r"D:\GraduationProject\demo1\output\align_eeg_pose.csv"

    # align_eeg_with_keypoints(eeg_file, keypoints_file, output_file)

    eeg_dir = r"D:\dataset\eeg_csv"

    kp_dir = r"D:\dataset\pose_feature"

    output_dir = r"C:\Users\dengm\Desktop\dataset\align_eeg_pose"
    # 在这里指定帧率
    VIDEO_FPS = 10
    batch_process(eeg_dir, kp_dir, output_dir,VIDEO_FPS)
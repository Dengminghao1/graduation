
import pandas as pd
import numpy as np
import os
import re


def align_eeg_with_facial_data(eeg_file, facial_file, output_file):
    """
    将脑电数据与面部识别数据按照时间对齐

    参数:
    eeg_file: 脑电数据CSV文件路径
    facial_file: 面部识别数据CSV文件路径
    output_file: 输出文件路径
    """

    # 1. 读取脑电数据
    print("正在读取脑电数据...")
    eeg_data = pd.read_csv(eeg_file)

    # 处理可能的BOM字符
    if eeg_data.columns[0].startswith('﻿'):
        eeg_data = eeg_data.rename(columns={eeg_data.columns[0]: eeg_data.columns[0].replace('﻿', '')})

    # 确保列名正确
    eeg_data.columns = ['timestamp', 'attention']

    # 将timestamp转换为datetime格式
    eeg_data['timestamp'] = pd.to_datetime(eeg_data['timestamp'])

    print(f"脑电数据记录数: {len(eeg_data)}")
    print(f"脑电数据时间范围: {eeg_data['timestamp'].min()} 到 {eeg_data['timestamp'].max()}")

    # 2. 读取面部识别数据
    print("\n正在读取面部识别数据...")
    facial_data = pd.read_csv(facial_file)

    # 清理列名中的空格
    facial_data.columns = [col.strip() for col in facial_data.columns]

    # 将timestamp列重命名为timestamp_seconds以避免混淆
    facial_data = facial_data.rename(columns={'timestamp': 'timestamp_seconds'})

    # 将actual_time转换为datetime格式
    facial_data['actual_time'] = pd.to_datetime(facial_data['actual_time'])

    print(f"面部识别数据记录数: {len(facial_data)}")
    print(f"面部识别数据时间范围: {facial_data['actual_time'].min()} 到 {facial_data['actual_time'].max()}")

    # 3. 检查时间重叠情况
    eeg_start = eeg_data['timestamp'].min()
    eeg_end = eeg_data['timestamp'].max()
    facial_start = facial_data['actual_time'].min()
    facial_end = facial_data['actual_time'].max()

    print(f"\n时间重叠检查:")
    print(f"脑电数据时间范围: {eeg_start} 到 {eeg_end}")
    print(f"面部数据时间范围: {facial_start} 到 {facial_end}")

    # 计算时间重叠
    overlap_start = max(eeg_start, facial_start)
    overlap_end = min(eeg_end, facial_end)

    if overlap_start > overlap_end:
        print("警告: 两个数据集没有时间重叠!")
        return None

    print(f"时间重叠范围: {overlap_start} 到 {overlap_end}")

    # 4. 对齐数据 - 为每个面部识别时间点找到最接近的脑电数据点
    print("\n正在对齐数据...")

    # 创建一个字典来存储时间戳到attention的映射
    eeg_timestamps = eeg_data['timestamp'].values
    eeg_attention = eeg_data['attention'].values

    # 用于存储对齐后的attention值
    aligned_attention = []

    # 遍历每个面部识别时间点
    for idx, row in facial_data.iterrows():
        facial_time = row['actual_time']

        # 如果面部识别时间在脑电数据时间范围内
        if facial_time >= eeg_start and facial_time <= eeg_end:
            # 修正这里：将时间差计算为秒数
            # 方法1：使用列表推导式计算时间差（秒）
            time_diffs_seconds = []
            for eeg_time in eeg_timestamps:
                # 计算时间差的绝对秒数
                diff = abs((eeg_time - facial_time).total_seconds())
                time_diffs_seconds.append(diff)

            # 找到最小时间差的索引
            min_idx = np.argmin(time_diffs_seconds)
            min_time_diff_seconds = time_diffs_seconds[min_idx]

            # 如果时间差在1秒内，认为是有效匹配
            if min_time_diff_seconds < 1.0:
                aligned_attention.append(eeg_attention[min_idx])
            else:
                aligned_attention.append(None)  # 没有足够接近的匹配
        else:
            aligned_attention.append(None)  # 时间超出范围

    # 计算匹配成功率
    matched_count = sum(1 for x in aligned_attention if x is not None)
    total_count = len(aligned_attention)
    match_rate = matched_count / total_count * 100 if total_count > 0 else 0

    print(f"\n对齐结果:")
    print(f"总面部识别数据点: {total_count}")
    print(f"成功匹配的脑电数据点: {matched_count}")
    print(f"匹配成功率: {match_rate:.2f}%")

    # 6. 保存结果
    print("\n正在保存对齐结果...")

    # 创建输出数据 - 保留原始面部识别数据的所有列
    output_data = facial_data.copy()

    # 将对齐的脑电attention添加到最后一列
    output_data['eeg_attention'] = aligned_attention

    # 保存到文件
    output_data.to_csv(output_file, index=False)
    print("\n数据对齐完成!")

    # 显示一些统计信息
    print("\n脑电注意力水平统计:")
    attention_stats = output_data['eeg_attention'].value_counts(dropna=True)
    print(attention_stats)

    # 计算有脑电数据的比例
    valid_eeg_ratio = output_data['eeg_attention'].notna().sum() / len(output_data) * 100
    print(f"\n有脑电数据的面部识别帧比例: {valid_eeg_ratio:.2f}%")
    return output_data


def align_files_by_time(eeg_dir, facial_dir, output_dir):
    """
    批量对齐两个目录下的文件，根据文件名中的时间戳进行匹配
    """
    print("开始批量对齐文件...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有脑电文件和面部文件
    eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith('.csv')]
    facial_files = [f for f in os.listdir(facial_dir) if f.endswith('.csv')]

    print(f"找到 {len(eeg_files)} 个脑电文件")
    print(f"找到 {len(facial_files)} 个面部文件")

    # 创建文件名到时间戳的映射
    eeg_time_map = {}
    facial_time_map = {}

    # 解析脑电文件名的时间戳
    for f in eeg_files:
        # 直接在完整文件名中查找时间戳模式
        time_match = re.search(r'(\d{14}_\d{14})', f)
        if time_match:
            time_key = time_match.group(1)
            eeg_time_map[time_key] = os.path.join(eeg_dir, f)

    # 解析面布文件名的时间戳
    for f in facial_files:
        # 直接在完整文件名中查找时间戳模式
        time_match = re.search(r'(\d{14}_\d{14})', f)
        if time_match:
            time_key = time_match.group(1)
            facial_time_map[time_key] = os.path.join(facial_dir, f)

    print(f"成功解析 {len(eeg_time_map)} 个脑电文件的时间戳")
    print(f"成功解析 {len(facial_time_map)} 个面部文件的时间戳")

    # 对齐匹配的文件
    matched_count = 0

    for time_key in eeg_time_map:
        if time_key in facial_time_map:
            print(f"\n匹配到文件对: {time_key}")
            matched_count = matched_count + 1
            eeg_path = eeg_time_map[time_key]
            facial_path = facial_time_map[time_key]
            # 构造输出文件名
            output_filename = f"align_eeg_face_{time_key}.csv"
            output_filepath = os.path.join(output_dir, output_filename)
            # 对齐单个文件对
            align_eeg_with_facial_data(eeg_path, facial_path, output_filepath)


    print(f"\n{'=' * 60}")
    print(f"批量对齐完成!")
    print(f"总匹配文件对: {matched_count}")
    print(f"输出目录: {output_dir}")

# 主程序
if __name__ == "__main__":
    # 文件路径
    # facial_file_path = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\face_feature_second\per_second_192.168.0.101_01_20231229150516_20231229151709.csv"  # 脑电数据文件
    # eeg_file_path = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs_csv\2021214387_周婉婷_20231229150516_20231229151709.csv"  # 面部识别数据文件
    # output_file_path = r"D:\GraduationProject\demo1\output\aligned_data.csv"  # 输出文件
    # 设置目录路径
    eeg_directory = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs_csv"  # 脑电数据目录
    facial_directory = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\face_feature_second"  # 面部数据目录
    output_directory = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\align_face_eeg_second"  # 输出目录

    # 执行批量对齐
    align_files_by_time(eeg_directory, facial_directory, output_directory)



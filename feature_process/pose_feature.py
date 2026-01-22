import glob
import os
import json
import csv
from pathlib import Path

import pandas as pd


def process_single_folder(folder_path, output_csv):
    """处理单个文件夹中的 JSON 文件并合并为 CSV"""
    # 获取并排序所有 json 文件
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    if not json_files:
        return False

    all_data = []
    max_keypoints = 0

    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if data.get("people") and len(data["people"]) > 0:
                    keypoints = data["people"][0].get("pose_keypoints_2d", [])
                    # 每一行：文件名, 关键点1, 关键点2...
                    all_data.append([file_name] + keypoints)
                    max_keypoints = max(max_keypoints, len(keypoints))
                else:
                    # 如果没人，只填文件名
                    all_data.append([file_name])
            except Exception as e:
                print(f"  [错误] 无法读取文件 {file_name}: {e}")

    # 生成表头
    header = ['file_name']
    for i in range(max_keypoints // 3):
        header.extend([f'x{i}', f'y{i}', f'conf{i}'])

    # 写入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)
    return True


def batch_process_all_folders(root_dir, output_root):
    """遍历父目录，对每个子文件夹执行操作"""
    # 确保输出目录存在
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 遍历 root_dir 下的第一层子目录
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    print(f"找到 {len(subfolders)} 个子文件夹，准备开始处理...\n" + "-" * 50)

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        output_csv_path = os.path.join(output_root, f"{folder_name}.csv")

        print(f"正在处理文件夹: {folder_name} ...", end="", flush=True)

        success = process_single_folder(folder, output_csv_path)

        if success:
            print(f" [完成] -> 已保存至 {os.path.basename(output_csv_path)}")
        else:
            print(f" [跳过] -> 文件夹内无 JSON 文件")

    print("-" * 50 + "\n所有任务处理完毕！")





# --- 参数配置 ---
# 1. 包含多个文件夹的父目录路径
input_parent_dir = r"C:\Users\dengm\Desktop\fsdownload\pose_10pfs"  # 替换为你的 JSON 文件夹路径
# 2. 存放生成的多个 CSV 文件的目标目录
output_save_dir = r"C:\Users\dengm\Desktop\fsdownload\pose_feature_merged"
# 执行批量处理
batch_process_all_folders(input_parent_dir, output_save_dir)

# process_single_folder(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\pose_feature\192.168.0.101_01_20231229172018_20231229173018",'192.168.0.101_01_20231229172018_20231229173018.csv')

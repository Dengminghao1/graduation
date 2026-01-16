import glob
import os

import pandas as pd


def merge_all_csvs_to_one(csv_folder, final_output_name):
    """
    将指定目录下所有的 CSV 文件合并为一个
    csv_folder: 存放各个子 CSV 的目录
    final_output_name: 合并后的总文件名
    """
    # 1. 获取目录下所有的 CSV 文件路径
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    if not csv_files:
        print("没有找到可合并的 CSV 文件。")
        return

    print(f"开始合并 {len(csv_files)} 个文件...")

    combined_list = []

    for file in csv_files:
        # 读取单个 CSV
        df = pd.read_csv(file)

        combined_list.append(df)

    # 2. 使用 pandas 快速合并
    # ignore_index=True 重新排列行索引
    master_df = pd.concat(combined_list, axis=0, ignore_index=True)

    # 3. 保存最终文件
    master_df.to_csv(final_output_name, index=False, encoding='utf-8-sig')

    print(f"--- 合并完成 ---")
    print(f"总行数: {len(master_df)}")
    print(f"最终文件保存至: {final_output_name}")


if __name__ == "__main__":
    merge_all_csvs_to_one(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\align_eeg_pose",
                         r"D:\GraduationProject\demo1\output\merged_pose_eeg_feature_files.csv")
import os

import pandas as pd


def merge_csv_by_time(csv1_path, csv2_path, output_path,
                      time_col1='actual_time',
                      time_col2='timestamp_pose'):
    """
    将两个CSV文件按时间列对齐合并

    参数：
        csv1_path: 第一个CSV文件路径（包含 actual_time）
        csv2_path: 第二个CSV文件路径（包含 timestamp_pose）
        output_path: 输出文件路径
        time_col1: 第一个文件的时间列名
        time_col2: 第二个文件的时间列名
    """

    # 读取CSV文件
    print("正在读取文件...")
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    print(f"文件{os.path.basename(csv1_path)}行数: {len(df1)}, "
          f"文件{os.path.basename(csv2_path)}行数: {len(df2)}")
    # 检查是否存在重复时间戳
    print("df1重复时间戳:", df1[time_col1].duplicated().sum())
    print("df2重复时间戳:", df2[time_col2].duplicated().sum())
    # 如需去重，可在合并前处理
    df1 = df1.drop_duplicates(subset=[time_col1])
    df2 = df2.drop_duplicates(subset=[time_col2])
    # 将时间列转换为数值（去除可能的空格）
    df1[time_col1] = pd.to_datetime(df1[time_col1])
    df2[time_col2] = pd.to_datetime(df2[time_col2])

    # 删除时间列为空的行
    df1 = df1.dropna(subset=[time_col1])
    df2 = df2.dropna(subset=[time_col2])

    # 执行内连接合并（只保留两个文件都有的时间点）
    merged_df = pd.merge(
        df1,
        df2,
        left_on=time_col1,
        right_on=time_col2,
        how='inner'  # 'inner'=交集, 'outer'=并集, 'left'=以df1为主, 'right'=以df2为主
    )

    # 可选：删除重复的时间列
    if time_col1 != time_col2 and time_col2 in merged_df.columns:
        merged_df = merged_df.drop(columns=[time_col2])

    print(f"合并后行数: {len(merged_df)}")

    # 保存结果
    merged_df.to_csv(output_path, index=False)
    print(f"已保存到: {output_path}")

    return merged_df


# 使用示例
if __name__ == "__main__":
    merge_csv_by_time(
        csv1_path=r"D:\GraduationProject\demo1\output\merged_face_eeg_feature_files.csv",  # 包含 actual_time 的文件
        csv2_path=r"D:\GraduationProject\demo1\output\merged_pose_eeg_feature_files.csv",  # 包含 timestamp_pose 的文件
        output_path=r"D:\GraduationProject\demo1\output\merged_face_pose_eeg_feature_files.csv",
        time_col1='actual_time',
        time_col2='timestamp_pose'
    )

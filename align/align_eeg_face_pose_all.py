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


def merge_and_check_time(f1, f2, t1, t2, output):
    """
    将两个CSV按行拼接，检查时间列是否一致，输出到新文件
    """
    try:
        # 1. 读取两个文件
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)

        # 2. 检查列名是否存在
        if t1 not in df1.columns:
            print(f"❌ 错误: 文件1中找不到列名 '{t1}'")
            print(f"  文件1的列名: {list(df1.columns)}")
            return
        if t2 not in df2.columns:
            print(f"❌ 错误: 文件2中找不到列名 '{t2}'")
            print(f"  文件2的列名: {list(df2.columns)}")
            return

        # 3. 按行拼接（横向拼接）
        merged_df = pd.concat([df1, df2], axis=1)

        # 4. 检查时间是否相同
        # 将时间列转换为 datetime 格式
        df1_time = pd.to_datetime(merged_df[t1], errors='coerce')
        df2_time = pd.to_datetime(merged_df[t2], errors='coerce')

        # 判断是否相等，新增一列
        merged_df['is_time_match'] = (df1_time == df2_time)

        # 5. 保存到新文件
        merged_df.to_csv(output, index=False)

        # 6. 统计并输出结果
        total_rows = len(merged_df)
        match_count = merged_df['is_time_match'].sum()
        mismatch_count = total_rows - match_count

        print(f"\n{'=' * 40}")
        print(f"✅ 拼接完成！")
        print(f"输出文件: {output}")
        print(f"总行数: {total_rows}")
        print(f"时间匹配: {match_count} 行")
        print(f"时间不匹配: {mismatch_count} 行")
        print(f"匹配率: {match_count / total_rows * 100:.2f}%")
        print(f"{'=' * 40}\n")

    except Exception as e:
        print(f"❌ 发生错误: {e}")


# 使用示例
if __name__ == "__main__":
    # merge_csv_by_time(
    #     csv1_path=r"C:\Users\dengm\Desktop\dataset\merged_face_eeg_feature_files.csv",  # 包含 actual_time 的文件
    #     csv2_path=r"C:\Users\dengm\Desktop\dataset\merged_pose_eeg_feature_files.csv",  # 包含 timestamp_pose 的文件
    #     output_path=r"C:\Users\dengm\Desktop\dataset\merged_face_pose_eeg_feature_files.csv",
    #     time_col1='actual_time',
    #     time_col2='timestamp_pose'
    # )
    # import pandas as pd

    # ================= 配置区域 =================
    # 两个文件路径
    file1_path = r"C:\Users\dengm\Desktop\dataset\merged_face_eeg_feature_files.csv"
    file2_path = r"C:\Users\dengm\Desktop\dataset\merged_pose_eeg_feature_files.csv"
    output_path = r"C:\Users\dengm\Desktop\dataset\merged_face_pose_eeg_feature_files.csv"
    # 定义两个文件中对应的时间列名
    time_col1 = 'actual_time'  # file1 中的时间列
    time_col2 = 'timestamp_pose'  # file2 中的时间列
    # 运行
    merge_and_check_time(file1_path, file2_path, time_col1, time_col2, output_path)

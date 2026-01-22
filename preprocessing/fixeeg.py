import pandas as pd
import os
from datetime import timedelta

# 原始CSV目录
input_folder = r"C:\Users\dengm\Desktop\eeg\eeg_csv"
# 输出目录
output_folder = r"C:\Users\dengm\Desktop\eeg\new_eeg_csv_1"
os.makedirs(output_folder, exist_ok=True)

# 遍历所有CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # 转 datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 构建完整时间序列
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        full_index = pd.date_range(start=start_time, end=end_time, freq='S')

        # 创建完整DataFrame
        df_full = pd.DataFrame({'timestamp': full_index})
        df_full = df_full.merge(df, on='timestamp', how='left')

        # 记录修复信息
        fix_records = []

        # 填充缺失时间：取上一行的值
        for i in range(len(df_full)):
            if pd.isna(df_full.loc[i, 'attention']):
                if i == 0:
                    df_full.loc[i, 'attention'] = '中性'  # 第一行缺失默认值
                    fix_records.append(f"{df_full.loc[i, 'timestamp']} 缺失 -> 填充默认值 中性")
                else:
                    prev_val = df_full.loc[i - 1, 'attention']
                    df_full.loc[i, 'attention'] = prev_val
                    fix_records.append(f"{df_full.loc[i, 'timestamp']} 缺失 -> 填充上一行值 {prev_val}")

        # 处理重复时间
        duplicates = df_full[df_full.duplicated(subset='timestamp', keep=False)]
        if not duplicates.empty:
            for t in duplicates['timestamp'].unique():
                dup_idx = df_full.index[df_full['timestamp'] == t].tolist()
                # 保留第一条，删除其余重复行
                for idx in dup_idx[1:]:
                    fix_records.append(f"{df_full.loc[idx, 'timestamp']} 重复 -> 删除")
                    df_full.drop(idx, inplace=True)

        df_full = df_full.sort_values('timestamp').reset_index(drop=True)

        # 保存修复后的CSV到新目录
        output_path = os.path.join(output_folder, filename)
        df_full.to_csv(output_path, index=False)

        # 输出修复日志
        if fix_records:
            print(f"文件 {filename} 修复记录：")
            for record in fix_records:
                print("  ", record)
        else:
            print(f"文件 {filename} 无需修复")

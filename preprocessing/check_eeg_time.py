import os
import pandas as pd

# ================= 配置区域 =================
# 请修改为你的目录路径
dir_path = r"D:\dataset\eeg_csv"

# 如果你的 CSV 文件第一行是标题（如 Time, Label），设为 0
# 如果第一行就是数据，设为 None
csv_header = 0

# ===========================================

def parse_filename(filename):
    """
    解析文件名，提取开始时间和结束时间
    格式示例: 2021214387_周婉婷_20231229150516_20231229151709.csv
    """
    try:
        name = os.path.splitext(filename)[0]
        parts = name.split('_')

        # parts[2] 是开始时间字符串, parts[3] 是结束时间字符串
        s_time_str = parts[2]
        e_time_str = parts[3]

        # 转换为 datetime 对象
        s_time = pd.to_datetime(s_time_str, format='%Y%m%d%H%M%S')
        e_time = pd.to_datetime(e_time_str, format='%Y%m%d%H%M%S')
        return s_time, e_time
    except Exception as e:
        return None, None


def check_file_integrity(filepath, start_time, end_time, header):
    """检查单个文件的时间完整性，返回详细信息"""
    try:
        # 读取文件，只读第一列
        df = pd.read_csv(filepath, header=header, usecols=[0], names=['time'])
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna()  # 去除无效格式的时间

        if df.empty:
            return [], []  # 无有效数据，返回空列表

        # 统一精确到秒 (去除毫秒)
        df['time_sec'] = df['time'].dt.floor('S')

        # 生成理论时间轴
        full_range = pd.date_range(start=start_time, end=end_time, freq='S')
        full_set = set(full_range)

        # 实际数据集合
        data_set = set(df['time_sec'])

        # 1. 统计缺失 - 具体列出缺失的时间点
        missing_times = sorted(list(full_set - data_set))

        # 2. 统计重复 - 具体列出重复的时间点和重复次数
        vc = df['time_sec'].value_counts()
        duplicated = vc[vc > 1]
        duplicate_details = []
        for t, count in duplicated.items():
            duplicate_details.append(f"{t} ({count}次)")

        return missing_times, duplicate_details

    except Exception as e:
        print(f"读取文件出错 {filepath}: {e}")
        return None, None


def batch_check_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    print(f"正在扫描目录: {directory}")
    print(f"共发现 {len(files)} 个 CSV 文件\n")

    for f in files:
        filepath = os.path.join(directory, f)
        s_time, e_time = parse_filename(f)

        if s_time and e_time:
            missing_times, dup_details = check_file_integrity(filepath, s_time, e_time, csv_header)

            if missing_times is None:
                print(f"⚠️ 文件读取错误: {f}")
                continue

            # 计算理论总数
            total_secs = int((e_time - s_time).total_seconds()) + 1

            print(f"{'=' * 60}")
            print(f"文件: {f}")
            print(f"时间范围: {s_time} 至 {e_time} (理论总数: {total_secs}秒)")
            print(f"{'-' * 60}")

            # 输出缺失详情
            if len(missing_times) > 0:
                print(f"⚠️ 缺失 {len(missing_times)} 个时间点:")
                for t in missing_times:
                    print(f"  - {t}")
            else:
                print("✅ 无缺失")

            # 输出重复详情
            if len(dup_details) > 0:
                print(f"⚠️ 重复 {len(dup_details)} 个时间点:")
                for detail in dup_details :
                    print(f"  - {detail}")
            else:
                print("想要更多细节?")

            print(f"{'=' * 60}\n")
        else:
            print(f"⚠️ 文件名格式无法解析: {f}")

    print("扫描完成。")


# 运行
batch_check_directory(dir_path)

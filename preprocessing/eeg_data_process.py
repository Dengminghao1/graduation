import os
from datetime import datetime
import re


def classify_attention_level(value):
    """根据数值返回注意力等级"""
    value = int(value)

    if value == 0:
        return "无效"
    elif 1 <= value <= 20:
        return "低"
    elif 20 < value <= 40:
        return "稍低"
    elif 40 < value <= 60:
        return "中性"
    elif 60 < value <= 80:
        return "稍高"
    elif 80 < value <= 100:
        return "高"
    else:
        return "异常值"  # 处理超出范围的情况


def process_file(input_file, output_file):
    """处理文件，提取时间和注意力等级"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 分割数据
        parts = line.split(',')

        # 提取时间（第一部分）
        timestamp = parts[0]

        # 提取倒数第二列（假设列数固定）
        attention_value = parts[-2] if len(parts) >= 2 else "0"

        # 分类
        level = classify_attention_level(attention_value)

        # 组合结果
        results.append(f"{timestamp},{level}")

    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("时间,注意力等级\n")
        f.write("\n".join(results))

    print(f"处理完成！共处理 {len(results)} 行数据")
    print(f"输出文件: {output_file}")
    # 统计各等级数量
    from collections import Counter
    levels = [line.split(',')[1] for line in results]
    counter = Counter(levels)

    print("\n等级分布统计:")
    for level, count in counter.items():
        print(f"  {level}: {count} 行 ({count / len(results) * 100:.1f}%)")


def filter_time_range(input_file_path, output_dir_path, start_time_str, end_time_str):
    """
    筛选指定时间范围内的数据行

    参数:
        input_file_path: 输入文件路径
        output_file_path: 输出文件路径
        start_time_str: 开始时间字符串 (格式: "YYYYmmddHHMMSS")
        end_time_str: 结束时间字符串 (格式: "YYYYmmddHHMMSS")
    """
    # 1. 将传入的时间字符串解析为对象，方便比较
    start_dt = datetime.strptime(start_time_str, '%Y%m%d%H%M%S')
    end_dt = datetime.strptime(end_time_str, '%Y%m%d%H%M%S')
    matched_count = 0

    print(f"正在读取文件: {input_file_path}")
    print(f"筛选范围: {start_time_str} ~ {end_time_str}")
    base_name = os.path.basename(input_file_path)
    filename_no_ext = os.path.splitext(base_name)[0]
    time_range_str = f"{start_dt.strftime('%Y%m%d%H%M%S')}-{end_dt.strftime('%Y%m%d%H%M%S')}"

    # 拼接完整输出路径: 原路径/原文件名_开始时间-结束时间.txt
    output_file_path = os.path.join(output_dir_path, f"{filename_no_ext}_{time_range_str}.txt")
    # 2. 打开文件进行读写
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
            open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # 去掉首尾空白字符（防止空行干扰）
            line = line.strip()
            if not line:
                continue

            # 3. 提取时间部分
            # 逻辑：时间前面可能没有其他东西，时间后面紧跟逗号。
            # split(',', 1) 表示只按第一个逗号分割一次，效率更高。
            parts = line.split(',', 1)

            if len(parts) < 2:
                continue  # 跳过格式不正确的行

            current_time_str = parts[0]

            try:
                # 4. 将当前行的时间字符串转为对象
                current_dt = datetime.strptime(current_time_str, '%Y-%m-%d %H:%M:%S')

                # 5. 判断是否在时间范围内 [开始, 结束]
                if start_dt <= current_dt <= end_dt:
                    # 写入输出文件，并换行
                    outfile.write(line + '\n')
                    matched_count += 1

            except ValueError:
                # 如果时间格式不对，跳过该行
                continue

    print(f"处理完成！共筛选出 {matched_count} 条数据。")
    print(f"结果已保存至: {output_file_path}")


def batch_filter_eegs_by_videos(video_dir_path, eeg_file_path, output_eeg_dir_path):
    """
    根据视频目录中的文件名（包含时间范围），筛选总日志文件的数据。

    参数:
        video_dir_path: 视频文件所在目录
        eeg_file_path: 原始的脑电文件路径
        output_eeg_dir_path: 筛选后的脑电文件保存目录
    """

    # 0. 检查输入文件是否存在
    if not os.path.exists(eeg_file_path):
        print(f"✗ 错误：找不到日志文件 - {eeg_file_path}")
        return

    # 0. 检查并创建输出目录
    if not os.path.exists(output_eeg_dir_path):
        os.makedirs(output_eeg_dir_path)
        print(f"提示：已创建输出目录 - {output_eeg_dir_path}")

    print(f"正在读取总日志文件: {eeg_file_path}")

    # ---------------------------------------------------------
    # 第一步：预读取总日志文件到内存
    # ---------------------------------------------------------
    # 如果日志文件非常大（超过几百兆），建议不要全部读入内存，
    # 但对于几万行的考试日志，全部读入内存是筛选效率最高的方法。

    log_entries = []
    try:
        with open(eeg_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析每一行的时间
                # 格式: 2023-12-29 15:05:16,data...
                parts = line.split(',', 1)
                if len(parts) >= 2:
                    try:
                        current_dt = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
                        # 将解析后的时间对象和原始行内容存入列表
                        log_entries.append((current_dt, line))
                    except ValueError:
                        # 忽略时间格式错误的行
                        continue
    except Exception as e:
        print(f"✗ 读取日志文件失败: {e}")
        return

    print(f"日志读取完毕，共 {len(log_entries)} 条有效数据。")
    print("-" * 50)

    # ---------------------------------------------------------
    # 第二步：遍历视频文件，提取时间并筛选日志
    # ---------------------------------------------------------

    video_count = 0

    for filename in os.listdir(video_dir_path):
        video_file_path = os.path.join(video_dir_path, filename)

        # 跳过子目录
        if not os.path.isfile(video_file_path):
            continue

        # 1. 正则提取文件名中的时间范围
        # 匹配格式: _YYYYmmddHHMMSS_YYYYmmddHHMMSS (例如: _20231229150516_20231229151000)
        match = re.search(r'_(\d{14})_(\d{14})', filename)

        if not match:
            # 如果文件名不包含时间标记，跳过
            # print(f"⚠ 跳过: {filename} (无时间标记)")
            continue

        start_time_str = match.group(1)
        end_time_str = match.group(2)

        # 2. 将字符串转为 datetime 对象
        try:
            start_dt = datetime.strptime(start_time_str, '%Y%m%d%H%M%S')
            end_dt = datetime.strptime(end_time_str, '%Y%m%d%H%M%S')
        except ValueError:
            print(f"✗ 跳过: {filename} (时间格式解析失败)")
            continue

        # 3. 确定输出的日志文件名
        # 保持文件名一致，只是后缀改为 .txt (或者保留原文件名也行，看需求)
        # 这里假设输出文件名为: 视频文件名.txt (例如: video_202301-202302.txt)
        base_name = os.path.basename(eeg_file_path)
        filename_no_ext = os.path.splitext(base_name)[0]
        time_range_str = f"{start_dt.strftime('%Y%m%d%H%M%S')}_{end_dt.strftime('%Y%m%d%H%M%S')}"

        # 拼接完整输出路径: 原路径/原文件名_开始时间-结束时间.txt
        output_file_path = os.path.join(output_eeg_dir_path, f"{filename_no_ext}_{time_range_str}.txt")

        # 4. 筛选数据并写入
        matched_count = 0
        try:
            with open(output_file_path, 'w', encoding='utf-8') as out_f:
                for dt, line_content in log_entries:
                    # 判断日志时间是否在视频的时间范围内
                    if start_dt <= dt <= end_dt:
                        out_f.write(line_content + '\n')
                        matched_count += 1

            print(f"✅ [{filename}]")
            print(f"   筛选时间: {start_dt} ~ {end_dt}")
            print(f"   输出脑电: {output_file_path}")
            print(f"   匹配行数: {matched_count}")
            print("-" * 50)
            video_count += 1

        except Exception as e:
            print(f"✗ 写入文件失败 [{filename}]: {e}")

    print(f"\n全部完成！共处理了 {video_count} 个视频对应的日志数据。")

# 使用示例
if __name__ == "__main__":
    # process_file(r"E:\数据\脑电_重新解码\group1\2019214823_颜喜乐.txt", r"D:\GraduationProject\demo1\output\output_levels.csv")
    # filter_time_range(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\脑电\2021214387_周婉婷.txt", r"D:\GraduationProject\demo1\output", "20231229150516", "20231229151709")
    batch_filter_eegs_by_videos(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\新建文件夹", r"D:\GraduationProject\demo1\output\2021214387_周婉婷.txt", r"D:\GraduationProject\demo1\output\eegs")
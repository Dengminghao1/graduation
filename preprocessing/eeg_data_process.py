import glob
import os
from datetime import datetime
import re
import pandas as pd


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
        output_file_path = os.path.join(output_eeg_dir_path, f"{filename_no_ext}_{time_range_str}.csv")

        # 4. 筛选数据并写入
        matched_count = 0
        try:
            with open(output_file_path, 'w', encoding='utf-8') as out_f:
                if log_entries:  # 确保有数据
                    # 获取原始文件的表头信息，通常原始文件的第一行可能是表头
                    # 从原始文件中读取表头
                    with open(eeg_file_path, 'r', encoding='utf-8') as original_file:
                        first_line = original_file.readline().strip()
                        if first_line:
                            # 假设第一行是表头或数据格式说明
                            # 如果第一行是数据而不是表头，可以根据实际数据格式调整
                            out_f.write(first_line + '\n')  # 写入表头或格式说明
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


def txt_to_csv(input_dir=".", output_dir="csv_output",
               delimiter=None, encoding="utf-8",
               skip_errors=True):
    """
    将目录下的所有TXT文件转换为CSV格式

    参数:
    ----------
    input_dir : str
        输入目录路径，默认为当前目录
    output_dir : str
        输出目录路径，默认为"csv_output"
    delimiter : str or None
        分隔符，如果为None则自动检测（尝试逗号、制表符、空格）
    encoding : str
        文件编码，默认为"utf-8"
    skip_errors : bool
        遇到错误时是否跳过文件，默认为True

    返回:
    ----------
    list: 成功转换的文件列表
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有TXT文件
    txt_pattern = os.path.join(input_dir, "*.txt")
    txt_files = glob.glob(txt_pattern)

    if not txt_files:
        print(f"在目录 '{input_dir}' 中没有找到TXT文件")
        return []

    print(f"找到 {len(txt_files)} 个TXT文件:")
    for file in txt_files:
        print(f"  {os.path.basename(file)}")

    converted_files = []
    failed_files = []

    # 自动检测常见分隔符
    possible_delimiters = [',', '\t', ';', ' ', '|']

    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_filename)

        print(f"\n处理文件: {filename}")

        try:
            # 读取文件前几行来检测分隔符
            if delimiter is None:
                detected_delimiter = None
                with open(txt_file, 'r', encoding=encoding) as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip() if first_line else ""

                    # 测试每个可能的分隔符
                    for delim in possible_delimiters:
                        if delim in first_line:
                            # 检查分隔符是否一致
                            if second_line and delim in second_line:
                                # 检查分割后的列数是否一致
                                parts1 = first_line.split(delim)
                                parts2 = second_line.split(delim)
                                if len(parts1) > 1 and len(parts1) == len(parts2):
                                    detected_delimiter = delim
                                    break

                if detected_delimiter is None:
                    # 如果没有检测到明确的分隔符，尝试空格（可能是固定宽度）
                    print(f"  警告: 无法自动检测分隔符，尝试按空格分割")
                    detected_delimiter = None  # 让pandas按空格分割

                used_delimiter = detected_delimiter
            else:
                used_delimiter = delimiter

            # 读取TXT文件
            if used_delimiter is None:
                # 使用pandas读取，它会尝试自动检测分隔符
                df = pd.read_csv(txt_file, delimiter=None, engine='python',
                                 encoding=encoding, on_bad_lines='warn',header=0)
            else:
                # 使用指定的分隔符
                df = pd.read_csv(txt_file, delimiter=used_delimiter,
                                 encoding=encoding, on_bad_lines='warn',header=0)

            # 保存为CSV
            df.to_csv(csv_path, index=False, encoding='utf-8-sig',header=True)  # utf-8-sig防止Excel乱码

            # 统计信息
            print(f"  成功转换: {csv_filename}")
            print(f"  原始行数: {len(df)} 行")
            print(f"  列数: {len(df.columns)} 列")
            print(f"  列名: {list(df.columns)}")
            if used_delimiter:
                print(f"  使用的分隔符: {repr(used_delimiter)}")

            converted_files.append((filename, csv_filename, len(df), len(df.columns)))

        except Exception as e:
            error_msg = f"转换失败: {e}"
            print(f"  {error_msg}")
            if skip_errors:
                failed_files.append((filename, str(e)))
            else:
                raise

    # 输出总结报告
    print(f"\n{'=' * 60}")
    print("转换完成!")
    print(f"成功转换: {len(converted_files)} 个文件")

    if converted_files:
        print("\n成功转换的文件:")
        for original, converted, rows, cols in converted_files:
            print(f"  {original} → {converted} ({rows}行 × {cols}列)")

    if failed_files:
        print(f"\n失败的文件 ({len(failed_files)} 个):")
        for filename, error in failed_files:
            print(f"  {filename}: {error}")

    print(f"\n所有CSV文件已保存到: {os.path.abspath(output_dir)}")

    return converted_files
# 使用示例
if __name__ == "__main__":
     # filter_time_range(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\脑电\2021214387_周婉婷.txt", r"D:\GraduationProject\demo1\output", "20231229150516", "20231229151709")
    batch_filter_eegs_by_videos(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total", r"D:\GraduationProject\demo1\output\2021214387_周婉婷.txt", r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs")
    # txt_to_csv(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs", r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\eegs_csv")
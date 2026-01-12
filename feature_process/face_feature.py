import csv
import glob
import os
from collections import defaultdict
from datetime import datetime, timedelta


def find_best_per_second(input_file, output_file):
    # 从文件名解析基准时间
    try:
        # 去掉扩展名
        basename = os.path.basename(input_file)
        name_without_ext = os.path.splitext(basename)[0]

        # 按_分割
        parts = name_without_ext.split('_')

        if len(parts) < 4:
            raise ValueError(f"文件名格式不正确: {input_file}")

        # 第二个_到第三个_之间的部分（索引2）
        time_str = parts[2]

        # 解析时间字符串，格式：YYYYMMDDHHMMSS
        if len(time_str) == 14:
            file_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        else:
            raise ValueError(f"时间字符串格式不正确: {time_str}")
        base_time = file_time
        print(f"从文件名解析的基准时间: {base_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"解析文件名时间失败: {e}")
        print("将使用当前时间作为基准")
        base_time = datetime.now()
    # 按秒分组存储数据
    second_data = defaultdict(list)

    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 读取标题行
        # 去除字段名中的前后空格
        headers = [header.strip() for header in headers]
        # 找到关键字段的索引
        try:
            timestamp_idx = headers.index('timestamp')
            success_idx = headers.index('success')
            confidence_idx = headers.index('confidence')
        except ValueError as e:
            missing_fields = []
            expected_fields = ['timestamp', 'success', 'confidence']
            for field in expected_fields:
                if field not in headers:
                    missing_fields.append(field)

            print(f"缺少必要字段: {missing_fields}")
            print(f"CSV文件实际包含的字段: {headers}")
            return
        # 在timestamp列后插入"具体时间"列
        headers_with_time = headers[:timestamp_idx + 1] + ['actual_time'] + headers[timestamp_idx + 1:]
        # 读取所有行
        for row in reader:
            if len(row) < max(timestamp_idx, success_idx, confidence_idx) + 1:
                continue

            try:
                # 解析时间戳并获取秒数（取整数部分）
                timestamp = float(row[timestamp_idx])
                second = int(timestamp)

                # 检查是否成功
                success = int(row[success_idx])
                if success != 1:
                    continue

                # 获取置信度
                confidence = float(row[confidence_idx])
                # 计算具体时间
                time_delta = timedelta(seconds=timestamp)
                actual_time = base_time + time_delta

                # 存储数据：行内容 + 置信度
                second_data[second].append({
                    'row': row,
                    'confidence': confidence,
                    'timestamp': timestamp,
                    'actual_time': actual_time
                })

            except (ValueError, IndexError) as e:
                print(f"解析行时出错: {e}")
                continue

    # 对每秒的数据按置信度排序，取最高值
    best_rows = []
    for second in sorted(second_data.keys()):
        if second_data[second]:
            # 按置信度降序排序
            sorted_data = sorted(second_data[second],
                                 key=lambda x: x['confidence'],
                                 reverse=True)
            best_data = sorted_data[0]  # 置信度最高的数据

            # 如果需要，可以添加注释显示这是该秒的最佳数据
            best_row = best_data['row'].copy()
            # 插入位置：timestamp_idx+1（在timestamp列之后）
            best_row_with_time = (best_row[:timestamp_idx + 1] +
                                  [best_data['actual_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]] +
                                  best_row[timestamp_idx + 1:])
            best_rows.append(best_row_with_time)

    # 写入新文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入标题行
        writer.writerow(headers_with_time)
        # 写入最佳数据行
        writer.writerows(best_rows)

    print(f"处理完成！")
    print(f"原始数据共 {sum(len(v) for v in second_data.values())} 行成功数据")
    print(f"输出数据共 {len(best_rows)} 行（每秒置信度最高）")
    print(f"保存到: {output_file}")

    # 显示每秒钟的选择结果
    print("\n各秒选择的数据:")
    print("=" * 60)
    for second in sorted(second_data.keys()):
        if second_data[second]:
            best_data = max(second_data[second], key=lambda x: x['confidence'])
            print(f"第 {second} 秒:")
            print(f"  时间戳: {best_data['timestamp']:.3f}")
            print(f"  置信度: {best_data['confidence']:.2f}")
            print(f"  可用数据条数: {len(second_data[second])}")
            print("-" * 40)


def process_directory(input_dir=".", output_dir="output"):
    """
    处理目录中的所有CSV文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"在目录 '{input_dir}' 中没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for file in csv_files:
        print(f"  {os.path.basename(file)}")

    print(f"\n开始处理...")

    # 处理每个文件
    for input_file in csv_files:
        try:
            # 为每个文件生成输出文件名
            input_basename = os.path.basename(input_file)
            output_filename = f"per_second_{input_basename}"
            output_file = os.path.join(output_dir, output_filename)

            print(f"\n处理: {input_basename}")
            print(f"输出: {output_filename}")

            # 调用之前编写的处理单个文件的函数
            find_best_per_second(input_file, output_file)

        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"处理完成！")
    print(f"处理了 {len(csv_files)} 个文件")
    print(f"输出文件保存在: {output_dir}")
# 使用示例
if __name__ == "__main__":
    input_file = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\face_feature\192.168.0.101_01_20231229150516_20231229151709.csv"
    output_file = r"D:\GraduationProject\demo1\output\best_per_second_output.csv"

    find_best_per_second(input_file, output_file)
    # process_directory(input_dir=r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\face_feature",
    #                   output_dir=r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\face_feature_second")

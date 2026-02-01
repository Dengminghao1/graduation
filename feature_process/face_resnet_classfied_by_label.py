import os
import re
from datetime import datetime, timedelta
import shutil
from collections import defaultdict


def organize_frames_by_label(frame_folder, label_file, output_root):
    """
    根据标签文件将视频帧图片分类到不同的标签文件夹

    Args:
        frame_folder: 包含视频帧图片的文件夹路径
        label_file: 标签文件路径
        output_root: 输出根目录
    """

    # 标签到文件夹的映射
    label_to_folder = {
        '低': '0',
        '稍低': '1',
        '中性': '2',
        '稍高': '3',
        '高': '4'
    }

    # 确保输出目录存在
    for folder in label_to_folder.values():
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    # 读取标签数据
    print("正在读取标签数据...")
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == 0:  # 跳过第一行（列名）
                continue
            line = line.strip()
            if not line:
                continue
            time_str, label = line.split(',')
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            labels[dt] = label

    print(f"共读取 {len(labels)} 个标签记录")

    # 获取所有图片文件
    print("\n正在扫描图片文件...")
    image_files = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    print(f"找到 {len(image_files)} 个图片文件")

    # 解析图片文件名，提取信息
    processed_count = 0
    unprocessed_count = 0

    for img_file in image_files:
        try:
            # 解析文件名 frame_000000_192.168.0.101_01_20231229155003_20231229160003.jpg
            match = re.match(r'frame_(\d+)_([^_]+)_(\d+)_(\d+)_(\d+)\.jpg', img_file)
            if not match:
                print(f"无法解析文件名: {img_file}")
                unprocessed_count += 1
                continue

            frame_count = int(match.group(1))  # 帧号
            ip_address = match.group(2)  # IP地址
            camera_id = match.group(3)  # 摄像头ID
            start_time_str = match.group(4)  # 开始时间
            end_time_str = match.group(5)  # 结束时间

            # 解析开始时间 20231229155003 -> 2023-12-29 15:50:03
            start_time = datetime.strptime(start_time_str, '%Y%m%d%H%M%S')

            # 计算当前帧对应的时间点
            # 第0帧对应开始时间，每秒10帧
            seconds_offset = frame_count / 10.0
            current_time = start_time + timedelta(seconds=seconds_offset)

            # 找到最接近的标签（四舍五入到秒）
            current_time_rounded = current_time.replace(microsecond=0)

            if current_time_rounded in labels:
                label = labels[current_time_rounded]

                if label in label_to_folder:
                    target_folder = label_to_folder[label]

                    # 构建输出路径
                    src_path = os.path.join(frame_folder, img_file)
                    dst_path = os.path.join(output_root, target_folder, img_file)

                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    processed_count += 1

                    if processed_count % 1000 == 0:
                        print(f"已处理 {processed_count} 个文件...")
                else:
                    print(f"未知标签: {label}, 文件: {img_file}")
                    unprocessed_count += 1
            else:
                # 如果没有找到精确匹配的标签，尝试寻找最近的时间点
                found = False
                time_diff = timedelta(seconds=1)  # 允许1秒的误差

                for label_time, label in labels.items():
                    if abs(label_time - current_time_rounded) <= time_diff:
                        if label in label_to_folder:
                            target_folder = label_to_folder[label]
                            src_path = os.path.join(frame_folder, img_file)
                            dst_path = os.path.join(output_root, target_folder, img_file)
                            shutil.copy2(src_path, dst_path)
                            processed_count += 1
                            found = True
                            break

                if not found:
                    unprocessed_count += 1

        except Exception as e:
            print(f"处理文件 {img_file} 时出错: {e}")
            unprocessed_count += 1

    # 打印统计信息
    print(f"\n处理完成!")
    print(f"成功分类: {processed_count} 个文件")
    print(f"未处理: {unprocessed_count} 个文件")

    # 统计每个标签的数量
    print("\n各标签分类统计:")
    for label, folder in label_to_folder.items():
        folder_path = os.path.join(output_root, folder)
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            print(f"标签 '{label}' (文件夹 {folder}): {count} 个文件")

if __name__ == "__main__":
    frame_folder = r"D:\dataset\frame_picture\face_extracted_frames_124"
    label_file = r"D:\dataset\2021214398_张颖.csv"
    output_root = r"D:\dataset\frame_picture\classified_frames_face_124"
    organize_frames_by_label(
        frame_folder=frame_folder,
        label_file=label_file,
        output_root=output_root)
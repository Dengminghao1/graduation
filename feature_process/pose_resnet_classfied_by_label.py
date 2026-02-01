import os
import re
from datetime import datetime, timedelta
import shutil


def organize_frames_by_label(frame_root, label_file, output_root):
    """
    递归读取 frame_root 下所有图片，根据标签文件分类到 5 个文件夹
    """

    # 标签到文件夹映射
    label_to_folder = {
        '低': '0',
        '稍低': '1',
        '中性': '2',
        '稍高': '3',
        '高': '4'
    }

    # 创建输出目录
    for folder in label_to_folder.values():
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    # 读取标签文件
    print("正在读取标签数据...")
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            line = line.strip()
            if not line:
                continue

            time_str, label = line.split(',')
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            labels[dt] = label

    print(f"共读取 {len(labels)} 条标签记录")

    # 新文件名 pattern
    pattern_new = re.compile(
        r'([^_]+)_(\d+)_(\d{14})_(\d{14})_(\d+)_rendered\.png'
    )

    processed_count = 0
    unprocessed_count = 0

    print("\n开始递归扫描图片文件...")

    # 递归遍历所有子目录
    for root, _, files in os.walk(frame_root):
        for file in files:
            if not file.endswith('_rendered.png'):
                continue

            match = pattern_new.match(file)
            if not match:
                print(f"无法解析文件名: {file}")
                unprocessed_count += 1
                continue

            try:
                ip_address = match.group(1)
                camera_id = match.group(2)
                start_time_str = match.group(3)
                end_time_str = match.group(4)
                frame_count = int(match.group(5))

                # 解析开始时间
                start_time = datetime.strptime(start_time_str, '%Y%m%d%H%M%S')

                # 每秒 10 帧
                seconds_offset = frame_count / 10.0
                current_time = start_time + timedelta(seconds=seconds_offset)
                current_time_rounded = current_time.replace(microsecond=0)

                label = None

                # 精确匹配
                if current_time_rounded in labels:
                    label = labels[current_time_rounded]
                else:
                    # 允许 ±1 秒误差
                    for t, l in labels.items():
                        if abs(t - current_time_rounded) <= timedelta(seconds=1):
                            label = l
                            break

                if label and label in label_to_folder:
                    target_folder = label_to_folder[label]

                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(output_root, target_folder, file)

                    shutil.copy2(src_path, dst_path)
                    processed_count += 1

                    if processed_count % 1000 == 0:
                        print(f"已处理 {processed_count} 张图片")
                else:
                    unprocessed_count += 1

            except Exception as e:
                print(f"处理文件 {file} 出错: {e}")
                unprocessed_count += 1

    # 统计信息
    print("\n处理完成!")
    print(f"成功分类: {processed_count}")
    print(f"未处理: {unprocessed_count}")

    print("\n各类别统计:")
    for label, folder in label_to_folder.items():
        folder_path = os.path.join(output_root, folder)
        count = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
        print(f"{label} ({folder}): {count}")


if __name__ == "__main__":
    frame_root = r"D:\dataset\frame_picture\pose_101_224"
    label_file = r"D:\dataset\2021214387_周婉婷.csv"
    output_root = r"D:\dataset\frame_picture\classified_frames_pose_101_224"

    organize_frames_by_label(
        frame_root=frame_root,
        label_file=label_file,
        output_root=output_root
    )

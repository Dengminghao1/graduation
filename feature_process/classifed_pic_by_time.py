import os
import re
import shutil
from pathlib import Path
from collections import defaultdict


def copy_images_by_time_interval(source_folder, output_base_folder, preview=True):
    """
    根据文件名中的时间区间将图片复制到指定输出目录下的不同文件夹

    Args:
        source_folder (str): 源图片所在的文件夹路径
        output_base_folder (str): 输出基础目录（分类的子文件夹会创建在这里）
        preview (bool): True=仅预览，不复制； False=实际复制文件
    """
    source_dir = Path(source_folder)
    output_dir = Path(output_base_folder)

    if not source_dir.exists():
        print(f"❌ 源文件夹不存在: {source_folder}")
        return

    # 确保输出基础目录存在（如果是实际执行模式）
    if not preview:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_files = list(source_dir.rglob('*.jpg')) + list(source_dir.rglob('*.png'))

    if not image_files:
        print("❌ 源文件夹中没有找到图片文件")
        return

    print(f"📂 源目录: {source_dir}")
    print(f"📤 输出目录: {output_dir}")
    print(f"📊 共找到 {len(image_files)} 个图片文件\n")

    # 统计分组
    groups = defaultdict(list)
    # 正则：匹配最后两个时间戳
    # pattern = re.compile(r'frame_\d+_.+?_(\d{14})_(\d{14})')
    pattern = re.compile(r'.*?_(\d{14})_(\d{14})_.*?\.')

    for img_file in image_files:
        match = pattern.search(img_file.name)
        if match:
            start_time = match.group(1)
            end_time = match.group(2)
            # 子文件夹名格式：开始时间_结束时间
            subfolder_name = f"{start_time}_{end_time}"
            groups[subfolder_name].append(img_file)

    print("=" * 80)
    print(f"{'模式':<10} | {'目标子文件夹':<45} | {'文件数':<5}")
    print("-" * 80)

    # 遍历分组进行操作
    for subfolder_name, file_list in groups.items():
        # 组合输出基础路径和子文件夹名
        target_dir = output_dir / subfolder_name

        if not preview:
            # 创建目标子文件夹
            target_dir.mkdir(parents=True, exist_ok=True)

        for file_path in file_list:
            if preview:
                print(f"{'[预览]':<10} | {subfolder_name:<45} | 1")
            else:
                # 执行模式：复制文件
                dest_path = target_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest_path)
                except Exception as e:
                    print(f"❌ 复制失败 {file_path.name}: {e}")

    print("-" * 80)
    print("👀 预览结束（源文件未改动）" if preview else "✅ 复制完成！")


# ================= 使用示例 =================
if __name__ == "__main__":
    # 配置路径
    source_images = r'D:\dataset\frame_picture\pose_extracted_frames_101'  # 源文件夹
    output_folder = r"D:\dataset\frame_picture\classfied_by_time_pose_101"  # 你指定的输出文件夹

    # 第一步：预览（强烈建议先运行这一步）
    # print("===== 第一步：预览操作 =====")
    # copy_images_by_time_interval(source_images, output_folder, preview=True)

    # 第二步：确认无误后，复制这行代码运行实际操作
    print("\n===== 第二步：开始复制 =====")
    copy_images_by_time_interval(source_images, output_folder, preview=False)

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

# ================= 配置区域 =================
# 请修改为你的视频所在目录
video_dir = r"C:\Users\dengm\Desktop\dataset\blur_video\vi"

# 输出目录（默认会在原目录下创建一个 'fps_10' 文件夹存放处理后的视频）
output_dir = os.path.join(video_dir, 'fps_10')

# 目标帧率
target_fps = 10


# ===========================================

def adjust_video_framerate(input_dir, output_dir, target_fps=10):
    """
    遍历目录，将所有视频调整到指定帧率
    """
    if not os.path.exists(input_dir):
        print(f"❌ 错误：目录不存在: {input_dir}")
        return

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 已创建输出目录: {output_dir}")

    # 获取所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in video_extensions]

    if not video_files:
        print(f"⚠️ 目录中没有找到视频文件")
        return

    print(f"📁 找到 {len(video_files)} 个视频文件，开始处理...\n")

    success_count = 0
    fail_count = 0

    for filename in video_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            print(f"🔄 正在处理: {filename}")

            # 加载视频
            clip = VideoFileClip(input_path)

            # 调整帧率
            # 方法1: 使用 write_videofile 的 fps 参数进行重新采样
            # 方法2: 使用 clip.set_fps() 然后写入（通常 write_videofile 的 fps 参数更直接）

            # 这里使用 write_videofile 的 fps 参数
            # 注意：如果原视频帧率高于 10fps，会降采样；如果低于 10fps，会保持原帧率或插帧

            # 如果原视频帧率低于目标帧率，set_fps 会通过插值提升帧率，但可能不自然
            # 如果只是想"每秒取10帧"（即降采样），不需要 set_fps，只需要 write_videofile 的 fps 参数

            # 建议的简单做法：直接用 write_videofile 的 fps 参数
            clip.write_videofile(
                output_path,
                fps=target_fps,
                codec='libx264',  # 使用 H.264 编码
                audio_codec='aac',  # 音频编码
                logger=None  # 不显示详细日志
            )

            # 关闭释放内存
            clip.close()

            print(f"✅ 完成: {filename}")
            success_count += 1

        except Exception as e:
            print(f"❌ 失败: {filename}, 错误: {e}")
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"✅ 成功: {success_count} 个")
    fps_10_dir = os.path.join(video_dir, 'fps_10')
    print(f"⚠️ 失败: {fail_count} 个")
    print(f"📂 输出位置: {fps_10_dir}")
    print(f"{'=' * 50}")


def check_null_values(file_path):
    """
    检测CSV文件中的空值并输出详细报告
    """
    try:
        # 1. 读取文件
        df = pd.read_csv(file_path)
        print(f"✅ 成功读取文件: {file_path}")
        print(f"   文件大小: {len(df)} 行 × {len(df.columns)} 列\n")

        # 2. 检测空值（包括 NaN、None、空字符串 " "）
        # 将各种形式的空值统一处理为 NaN
        df_replaced = df.replace(r'^\s*$', np.nan, regex=True)

        # 3. 汇总统计
        total_cells = len(df) * len(df.columns)
        null_cells = df_replaced.isnull().sum().sum()
        null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 0

        print("=" * 60)
        print("【空值统计汇总】")
        print("=" * 60)
        print(f"总单元格数: {total_cells}")
        print(f"空值单元格数: {null_cells}")
        print(f"空值占比: {null_percentage:.2f}%")

        # 4. 每列的空值情况
        print("\n" + "=" * 60)
        print("【各列空值详情】")
        print("=" * 60)

        col_null_counts = df_replaced.isnull().sum()
        null_columns = col_null_counts[col_null_counts > 0]

        if len(null_columns) == 0:
            print("✅ 完美！没有发现空值。")
        else:
            print(f"\n共有 {len(null_columns)} 列存在空值:\n")
            print(f"{'列名':<30} {'空值数量':<10} {'空值占比':<10}")
            print("-" * 60)
            for col, count in null_columns.items():
                pct = count / len(df) * 100
                print(f"{col:<30} {count:<10} {pct:.2f}%")

        # 5. 显示有空值的行（前10行）
        null_rows = df_replaced[df_replaced.isnull().any(axis=1)]
        null_row_indices = null_rows.index.tolist()

        print("\n" + "=" * 60)
        print("【包含空值的行位置】")
        print("=" * 60)

        if len(null_rows) == 0:
            print("✅ 没有空值行")
        else:
            print(f"\n共有 {len(null_rows)} 行包含空值")
            print(f"空值行索引 (前20行): {null_row_indices[:20]}")
            if len(null_row_indices) > 20:
                print(f"                  ... 还有 {len(null_row_indices) - 20} 行")

            # 显示具体的空值位置
            print("\n具体空值位置 (前10行):")
            print("-" * 60)
            for idx in null_rows.head(10).index:
                null_cols_in_row = df_replaced.columns[df_replaced.loc[idx].isnull()].tolist()
                print(f"第 {idx} 行: 空值列 -> {null_cols_in_row}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

    # 迹行函数
    # adjust_video_framerate(video_dir, output_dir, target_fps)


def delete_columns(input_path, output_path, delete_names=None, delete_range=None):
    """
    删除CSV文件的列

    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        delete_names: 要删除的列名列表，如 ['col1', 'col2']
        delete_range: 要删除的列索引范围 (起始, 结束)，如 (2, 5)
    """
    try:
        # 1. 读取文件
        df = pd.read_csv(input_path)
        print(f"✅ 原始列数: {len(df.columns)}")
        print(f"   列名: {list(df.columns)}\n")

        cols_to_drop = []

        # 2. 按列名删除
        if delete_names:
            # 只删除实际存在的列
            existing_names = [col for col in delete_names if col in df.columns]
            if existing_names:
                cols_to_drop.extend(existing_names)
                print(f"🗑️  按名称删除: {existing_names}")
            else:
                print(f"⚠️  未找到指定的列名: {delete_names}")

        # 3. 按索引范围删除
        if delete_range:
            start, end = delete_range
            if start >= 0 and end < len(df.columns) and start <= end:
                # 获取该范围内的列名
                cols_by_index = list(df.columns[start:end + 1])
                cols_to_drop.extend(cols_by_index)
                print(f"🗑️  按索引删除 (第{start}列到第{end}列): {cols_by_index}")
            else:
                print(f"⚠️  索引范围无效: ({start}, {end})")

        # 4. 去重（避免按名称和按索引重复删除同一列）
        cols_to_drop = list(set(cols_to_drop))

        if cols_to_drop:
            # 5. 删除列
            df = df.drop(columns=cols_to_drop)
            print(f"\n✅ 实际删除: {cols_to_drop}")
            print(f"   剩余列数: {len(df.columns)}")
        else:
            print("\n⚠️  没有需要删除的列")

        # 6. 保存结果
        df.to_csv(output_path, index=False)
        print(f"\n💾 文件已保存到: {output_path}\n")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{input_path}'")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


def add_prefix_to_folders(directory):
    """
    遍历目录，给所有子文件夹添加指定前缀
    """
    prefix = '192.168.0.101_01_'
    if not os.path.exists(directory):
        print(f"❌ 错误：目录不存在: {directory}")
        return

    # 获取所有子文件夹
    folders = [f for f in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, f))]

    if not folders:
        print(f"⚠️ 目录中没有找到子文件夹")
        return

    print(f"📁 找到 {len(folders)} 个文件夹\n")

    success_count = 0
    fail_count = 0

    for folder_name in folders:
        # 如果已经有了前缀，跳过
        if folder_name.startswith(prefix):
            print(f"⏭️  已有前缀，跳过: {folder_name}")
            continue

        old_path = os.path.join(directory, folder_name)
        new_name = prefix + folder_name
        new_path = os.path.join(directory, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"✅ 重命名: {folder_name} → {new_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ 失败: {folder_name}, 错误: {e}")
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"✅ 成功: {success_count} 个")
    print(f"⚠️ 失败: {fail_count} 个")
    print(f"{'=' * 50}")


def check_null_values(file_path):
    """
    检测CSV文件中的空值并输出详细报告
    """
    try:
        # 1. 读取文件
        df = pd.read_csv(file_path)
        print(f"✅ 成功读取文件: {file_path}")
        print(f"   文件大小: {len(df)} 行 × {len(df.columns)} 列\n")

        # 2. 检测空值（包括 NaN、None、空字符串 " "）
        # 将各种形式的空值统一处理为 NaN
        df_replaced = df.replace(r'^\s*$', np.nan, regex=True)

        # 3. 汇总统计
        total_cells = len(df) * len(df.columns)
        null_cells = df_replaced.isnull().sum().sum()
        null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 0

        print("=" * 60)
        print("【空值统计汇总】")
        print("=" * 60)
        print(f"总单元格数: {total_cells}")
        print(f"空值单元格数: {null_cells}")
        print(f"空值占比: {null_percentage:.2f}%")

        # 4. 每列的空值情况
        print("\n" + "=" * 60)
        print("【各列空值详情】")
        print("=" * 60)

        col_null_counts = df_replaced.isnull().sum()
        null_columns = col_null_counts[col_null_counts > 0]

        if len(null_columns) == 0:
            print("✅ 完美！没有发现空值。")
        else:
            print(f"\n共有 {len(null_columns)} 列存在空值:\n")
            print(f"{'列名':<30} {'空值数量':<10} {'空值占比':<10}")
            print("-" * 60)
            for col, count in null_columns.items():
                pct = count / len(df) * 100
                print(f"{col:<30} {count:<10} {pct:.2f}%")

            # 5. 用0填充空值
            df_replaced = df_replaced.fillna(0)
            print(f"\n✅ 已将所有空值填充为0")

        # 6. 显示有空值的行（前10行）- 这里显示填充前的空值情况
        null_rows = df[df_replaced.isnull().any(axis=1)]  # 使用原始df显示空值情况
        null_row_indices = null_rows.index.tolist()

        print("\n" + "=" * 60)
        print("【包含空值的行位置】")
        print("=" * 60)

        if len(null_rows) == 0:
            print("✅ 没有空值行")
        else:
            print(f"\n共有 {len(null_rows)} 行包含空值")
            print(f"空值行索引 (前20行): {null_row_indices[:20]}")
            if len(null_row_indices) > 20:
                print(f"                  ... 还有 {len(null_row_indices) - 20} 行")

            # 显示具体的空值位置
            print("\n具体空值位置 (前10行):")
            print("-" * 60)
            for idx in null_rows.head(10).index:
                null_cols_in_row = df_replaced.columns[df.loc[idx].isnull()].tolist()
                print(f"第 {idx} 行: 空值列 -> {null_cols_in_row}")

        # 7. 保存填充后的数据
        filled_file_path = file_path.replace('.csv', '_filled.csv')
        df_replaced.to_csv(filled_file_path, index=False)
        print(f"\n💾 已将填充后的数据保存到: {filled_file_path}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


def check_frame_sequence(folder_path, pattern=r'frame_(\d{6})'):
    """
    检查文件夹中图片的帧序号是否连续

    Args:
        folder_path: 图片所在文件夹路径
        pattern: 用于匹配帧号的正则表达式

    Returns:
        打印详细的断帧报告
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 获取所有图片文件
    image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))

    if not image_files:
        print("❌ 文件夹中没有找到图片文件")
        return

    print(f"📁 正在检查文件夹: {folder_path}")
    print(f"📊 共找到 {len(image_files)} 个图片文件\n")

    # 用于存储每个视频序列的帧号
    # 结构: {视频唯一标识: [帧号列表]}
    video_frames = defaultdict(list)

    # 解析文件名
    for img_file in image_files:
        filename = img_file.name

        # 提取帧号 (例如: frame_000000_xxx.jpg -> 000000)
        match = re.search(pattern, filename)
        if match:
            frame_num = int(match.group(1))

            # 提取视频标识 (去掉帧号部分)
            # 例如: 192.168.0.101_01_20231229150516_20231229151709
            video_id = filename.replace(f"frame_{match.group(1)}_", "")
            # 去掉扩展名
            video_id = os.path.splitext(video_id)[0]

            video_frames[video_id].append(frame_num)

    # 检查每个视频序列的连续性
    all_continuous = True

    print("=" * 80)
    print("【断帧检测报告】")
    print("=" * 80)

    for video_id, frame_numbers in sorted(video_frames.items()):
        # 排序帧号
        frame_numbers_sorted = sorted(frame_numbers)
        total_frames = len(frame_numbers_sorted)

        print(f"\n📹 视频ID: {video_id}")
        print(f"   总帧数: {total_frames}")
        print(f"   帧号范围: {frame_numbers_sorted[0]} ~ {frame_numbers_sorted[-1]}")

        # 检查是否连续
        gaps = []
        for i in range(1, len(frame_numbers_sorted)):
            if frame_numbers_sorted[i] != frame_numbers_sorted[i - 1] + 1:
                gaps.append({
                    'previous': frame_numbers_sorted[i - 1],
                    'current': frame_numbers_sorted[i],
                    'gap_size': frame_numbers_sorted[i] - frame_numbers_sorted[i - 1] - 1
                })

        if gaps:
            all_continuous = False
            print(f"   ⚠️  发现 {len(gaps)} 处断帧:")
            for gap in gaps[:10]:  # 只显示前10处断帧，避免刷屏
                print(f"      从 {gap['previous']} 到 {gap['current']} 缺少 {gap['gap_size']} 帧")
            if len(gaps) > 10:
                print(f"      ... 还有 {len(gaps) - 10} 处断帧未显示")
        else:
            print(f"   ✅ 帧序号连续！")

    # 统计汇总
    total_videos = len(video_frames)
    continuous_videos = sum(1 for frames in video_frames.values()
                            if sorted(frames) == list(range(min(frames), max(frames) + 1)))

    print("\n" + "=" * 80)
    print("【汇总统计】")
    print("=" * 80)
    print(f"📊 检测的视频序列总数: {total_videos}")
    print(f"✅ 连续的视频序列: {continuous_videos}")
    print(f"❌ 有断帧的视频序列: {total_videos - continuous_videos}")
    print("=" * 80)

    return all_continuous


def delete_images_by_pattern(folder_path, pattern, preview=True):
    """
    删除文件名中包含特定子串的图片

    Args:
        folder_path (str): 图片所在文件夹路径
        pattern (str): 要匹配的子串（例如：192.168.0.124_01_20231229160026_20231229160416）
        preview (bool): True=仅预览，不删除； False=实际删除
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 获取所有图片文件
    image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))

    if not image_files:
        print("❌ 文件夹中没有找到图片文件")
        return

    print(f"📁 扫描文件夹: {folder_path}")
    print(f"🔍 匹配模式: {pattern}")
    print(f"📊 共找到 {len(image_files)} 个图片文件\n")

    # 筛选出要删除的文件
    files_to_delete = [f for f in image_files if pattern in f.name]

    if not files_to_delete:
        print(f"✅ 没有找到包含 '{pattern}' 的文件")
        return

    print("=" * 80)
    print(f"{'模式':<10} | {'文件名':<70}")
    print("-" * 80)

    # 遍历并处理
    for file_path in files_to_delete:
        if preview:
            print(f"{'[预览]':<10} | {file_path.name:<70}")
        else:
            try:
                file_path.unlink()  # 删除文件
                print(f"{'[删除]':<10} | {file_path.name:<70}")
            except Exception as e:
                print(f"{'[失败]':<10} | {file_path.name:<70} (原因: {e})")

    print("-" * 80)
    print(f"\n📊 统计: 共找到 {len(files_to_delete)} 个匹配文件")
    print("👀 预览结束（文件未删除）" if preview else "✅ 删除完成！")

if __name__ == '__main__':

    # # ================= 配置区域 =================
    # input_file = r"C:\Users\dengm\Desktop\dataset\merged_face_pose_eeg_feature_files_new2.csv"  # 输入文件路径
    # output_file = r"C:\Users\dengm\Desktop\dataset\merged_face_pose_eeg_feature_files_new3.csv"  # 输出文件路径
    #
    # # 方式1：按列名删除（列名列表）
    # delete_by_name = ['is_time_match'
    #                   ]
    #
    # # 方式2：按列索引范围删除（从0开始，包含起始和结束）
    # # 例如：(2, 5) 表示删除第3列到第6列（索引2到5）
    # delete_by_index = (696 - 1, 709 - 1)  # 设为 None 不使用此方式
    # # 运行示例
    # delete_columns(
    #     input_file,
    #     output_file,
    #     delete_names=delete_by_name,  # 按列名删除
    #     delete_range=delete_by_index  # 按索引范围删除
    # )
    # csv_file_path=r"C:\Users\dengm\Desktop\dataset\merged_face_pose_eeg_feature_files_new3_filled.csv"
    # # 运行检测
    # check_null_values(csv_file_path)

    # C:\Users\dengm\Desktop\dataset\frames\20231229150516_20231229151709
    # 修改为你的图片文件夹路径
    # image_folder = r"C:\Users\dengm\Desktop\dataset\frames\20231229150516_20231229151709"
    #
    # # 执行检查
    # check_frame_sequence(image_folder)
    image_folder = r"D:\A_from_ubuntu\extracted_frames_all\extracted_frames"  # 你的图片文件夹
    target_pattern = "192.168.0.124_01_20231229160026_20231229160416"  # 要删除的名字片段

    # 第一步：预览（强烈建议先运行这一步）
    print("===== 第一步：预览操作 =====")
    delete_images_by_pattern(image_folder, target_pattern, preview=False)
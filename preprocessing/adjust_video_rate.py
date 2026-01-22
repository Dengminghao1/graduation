import os
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


# 迹行函数
# adjust_video_framerate(video_dir, output_dir, target_fps)

import os
import shutil
if __name__ == '__main__':
    # # 原始图片目录
    # input_dir = r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\extracted_frames"
    #
    # # 整理后的输出目录
    # output_dir = r"C:\Users\dengm\Desktop\dataset\frames"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # for filename in os.listdir(input_dir):
    #     if not filename.lower().endswith(".jpg"):
    #         continue
    #
    #     # 去掉扩展名并拆分
    #     name = os.path.splitext(filename)[0]
    #     parts = name.split("_")
    #
    #     # 取最后两个字段作为时间区间
    #     start_time = parts[-2]
    #     end_time = parts[-1]
    #     time_folder = f"{start_time}_{end_time}"
    #
    #     # 创建目标目录
    #     target_dir = os.path.join(output_dir, time_folder)
    #     os.makedirs(target_dir, exist_ok=True)
    #
    #     # 复制文件
    #     src_path = os.path.join(input_dir, filename)
    #     dst_path = os.path.join(target_dir, filename)
    #     shutil.copy2(src_path, dst_path)  # copy2 会保留时间等元信息
    #
    # print("图片复制并整理完成 ✅")

    # # ================= 配置区域 =================
    # # 请修改为你的目录路径
    # target_dir = r'D:\dataset\extract_face_frames'
    # # ===========================================
    #
    # prefix = '192.168.0.101_01_'
    #
    #
    # def add_prefix_to_folders(directory):
    #     """
    #     遍历目录，给所有子文件夹添加指定前缀
    #     """
    #     if not os.path.exists(directory):
    #         print(f"❌ 错误：目录不存在: {directory}")
    #         return
    #
    #     # 获取所有子文件夹
    #     folders = [f for f in os.listdir(directory)
    #                if os.path.isdir(os.path.join(directory, f))]
    #
    #     if not folders:
    #         print(f"⚠️ 目录中没有找到子文件夹")
    #         return
    #
    #     print(f"📁 找到 {len(folders)} 个文件夹\n")
    #
    #     success_count = 0
    #     fail_count = 0
    #
    #     for folder_name in folders:
    #         # 如果已经有了前缀，跳过
    #         if folder_name.startswith(prefix):
    #             print(f"⏭️  已有前缀，跳过: {folder_name}")
    #             continue
    #
    #         old_path = os.path.join(directory, folder_name)
    #         new_name = prefix + folder_name
    #         new_path = os.path.join(directory, new_name)
    #
    #         try:
    #             os.rename(old_path, new_path)
    #             print(f"✅ 重命名: {folder_name} → {new_name}")
    #             success_count += 1
    #         except Exception as e:
    #             print(f"❌ 失败: {folder_name}, 错误: {e}")
    #             fail_count += 1
    #
    #     print(f"\n{'=' * 50}")
    #     print(f"处理完成！")
    #     print(f"✅ 成功: {success_count} 个")
    #     print(f"⚠️ 失败: {fail_count} 个")
    #     print(f"{'=' * 50}")
    #
    #
    # # 迥行函数
    # add_prefix_to_folders(target_dir)
    import pandas as pd



    # ===========================================

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


    # ================= 配置区域 =================
    input_file = r"C:\Users\dengm\Desktop\dataset\merged_face_pose_eeg_feature_files_new2.csv"  # 输入文件路径
    output_file = r"C:\Users\dengm\Desktop\dataset\merged_face_pose_eeg_feature_files_new3.csv"  # 输出文件路径

    # 方式1：按列名删除（列名列表）
    delete_by_name = ['is_time_match'
]

    # 方式2：按列索引范围删除（从0开始，包含起始和结束）
    # 例如：(2, 5) 表示删除第3列到第6列（索引2到5）
    delete_by_index = (696-1, 709-1)  # 设为 None 不使用此方式
    # 运行示例
    delete_columns(
        input_file,
        output_file,
        delete_names=delete_by_name,  # 按列名删除
        delete_range=delete_by_index  # 按索引范围删除
    )


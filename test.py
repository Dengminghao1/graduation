import cv2
from pathlib import Path
from typing import Dict


def count_video_frames(input_dir: str, video_extensions: list = None) -> Dict[str, int]:
    """
    统计目录下每个视频的帧数

    参数：
        input_dir: 输入目录路径
        video_extensions: 视频文件扩展名列表，默认常见格式

    返回：
        字典: {文件名: 帧数}
    """
    if video_extensions is None:
        video_extensions = ['.MP4', '.avi', '.mov', '.mkv', '.flv']

    input_path = Path(input_dir)
    video_frames = {}

    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))

    if not video_files:
        print(f"在目录 {input_dir} 中未找到视频文件")
        return video_frames

    print(f"找到 {len(video_files)} 个视频文件，开始统计帧数...\n")

    for video_file in video_files:
        try:
            cap = cv2.VideoCapture(str(video_file))

            if not cap.isOpened():
                print(f"❌ 无法打开: {video_file.name}")
                continue

            # 方法1: 直接读取总帧数属性（快速）
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 方法2: 如果上述方法不准，可以逐帧读取（慢但准确）
            # total_frames = 0
            # while True:
            #     ret, frame = cap.read()
            #     if not ret:
            #         break
            #     total_frames += 1

            cap.release()

            video_frames[video_file.name] = total_frames
            print(f"✓ {video_file.name:<40} {total_frames:>8} 帧")

        except Exception as e:
            print(f"❌ 处理 {video_file.name} 时出错: {e}")

    print(f"\n{'=' * 60}")
    print(f"统计完成! 共 {len(video_frames)} 个视频")
    print(f"{'=' * 60}")

    return video_frames


# 使用示例
if __name__ == "__main__":
    import torch

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
    result = count_video_frames(
        input_dir=r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total"
    )
    sum =0

    # 打印汇总
    if result:
        print("\n汇总结果:")
        for name, frames in result.items():
            sum += frames
            print(f"  {name}: {frames} 帧")
    print(sum)

import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time


def process_single_video(args):
    """处理单个视频（用于多进程）"""
    video_path, openpose_exe, output_dir = args
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)

    try:
        os.makedirs(video_output_dir, exist_ok=True)

        cmd = [
            openpose_exe,
            '--video', video_path,
            '--write_images', video_output_dir,  # 保存提取的姿态图片
            '--render_pose', '1',  # 渲染姿态（1=开启）
            '--display', '0',  # 关闭可视化窗口（0=关闭）
            '--disable_blending',  # 禁用混合（无值参数，直接加）
            '--tracking', '1',  # 开启人体跟踪（1=开启）
            '--number_people_max', '1'  # 限制最大检测人数为1
        ]

        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                cwd=r"C:\Users\dengm\Desktop\openpose")

        if result.returncode == 0:
            return (video_name, True, None)
        else:
            return (video_name, False, result.stderr[:500])

    except Exception as e:
        return (video_name, False, str(e))


def batch_process_parallel(input_dir, output_base_dir="output", openpose_exe=r".\bin\OpenPoseDemo.exe",
                           num_processes=None):
    """
    多进程批量处理视频

    参数:
    - input_dir: 输入视频目录
    - output_base_dir: 输出基础目录
    - openpose_exe: OpenPose可执行文件路径
    - num_processes: 进程数，默认使用CPU核心数
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']

    # 收集视频文件
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"❌ 没有找到视频文件")
        return

    print(f"📁 找到 {len(video_files)} 个视频文件")

    # 设置进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(video_files))
    print(f"⚙️ 使用 {num_processes} 个进程进行并行处理")

    # 准备参数
    args_list = [(video, openpose_exe, output_base_dir) for video in video_files]

    start_time = time.time()

    # 使用多进程池处理
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_video, args_list)

    # 统计结果
    success_count = sum(1 for _, success, _ in results if success)
    fail_count = len(results) - success_count

    # 打印结果
    print("\n" + "=" * 50)
    print("处理结果:")
    for video_name, success, error in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status}: {video_name}")
        if error:
            print(f"  错误: {error}")

    print("=" * 50)
    print(f"总计: {len(video_files)} 个视频")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {time.time() - start_time:.1f}秒")


# 使用示例
if __name__ == "__main__":
    batch_process_parallel(
        input_dir=r"C:\Users\dengm\Desktop\dataset\blur_video\video_101",
        output_base_dir=r"C:\Users\dengm\Desktop\dataset\blur_video\pose_101",
        openpose_exe=r"C:\Users\dengm\Desktop\openpose\bin\OpenPoseDemo.exe",
        num_processes=1  # 可以调整进程数
    )

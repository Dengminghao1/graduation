from pathlib import Path
import argparse
import cv2
import numpy as np
from mtcnn import MTCNN
import time


def align_face_mtcnn(image_path, output_size=224, margin=20, do_align=True):
    """
    使用MTCNN进行人脸检测、关键点定位和对齐

    参数：
    image_path: 输入图片路径
    output_size: 输出图片大小
    margin: 边界扩展像素
    do_align: 是否进行旋转对齐处理
              True: 进行旋转对齐（默认）
              False: 不旋转，只进行人脸检测和裁剪

    返回：
    aligned_face: 对齐后的人脸
    landmarks: 关键点坐标
    """

    # 初始化MTCNN检测器
    detector = MTCNN()

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测人脸
    results = detector.detect_faces(img_rgb)

    if len(results) == 0:
        print("未检测到人脸")
        return None, None

    # 获取第一个检测到的人脸
    result = results[0]
    bounding_box = result['box']
    landmarks = result['keypoints']

    # 提取关键点
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    # 在旋转对齐部分修改关键点变换计算
    if do_align:
        print("执行旋转对齐处理...")

        # 计算眼睛中心点
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)

        # 计算眼睛角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # 计算旋转矩阵
        center = (eyes_center[0], eyes_center[1])
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # 执行旋转
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_CUBIC)

        # 计算旋转后的人脸边界框
        # 使用原始边界框进行旋转，而不是关键点
        x, y, w, h = bounding_box
        # 定义人脸边界框的四个角点
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])

        # 添加齐次坐标
        ones = np.ones(shape=(len(corners), 1))
        corners_homogeneous = np.hstack([corners, ones])

        # 应用旋转矩阵
        rotated_corners = M.dot(corners_homogeneous.T).T

        # 计算旋转后的边界框
        x_min = int(rotated_corners[:, 0].min() - margin)
        x_max = int(rotated_corners[:, 0].max() + margin)
        y_min = int(rotated_corners[:, 1].min() - margin)
        y_max = int(rotated_corners[:, 1].max() + margin)

        # 确保边界在图像范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(rotated.shape[1], x_max)
        y_max = min(rotated.shape[0], y_max)

        # 裁剪旋转后的人脸
        aligned_face = rotated[y_min:y_max, x_min:x_max]

    else:
        # ========== 模式2：不进行旋转对齐 ==========
        print("不进行旋转对齐，只进行人脸裁剪...")

        # 直接使用MTCNN检测的边界框进行裁剪
        x, y, w, h = bounding_box

        # 扩展边界框
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(img.shape[1], x + w + margin)
        y_max = min(img.shape[0], y + h + margin)

        # 裁剪人脸区域
        aligned_face = img[y_min:y_max, x_min:x_max]

    # 调整到输出大小
    aligned_face = cv2.resize(aligned_face, (output_size, output_size))

    return aligned_face, landmarks


def process_video_to_frames(video_path, output_dir,
                            frame_interval=1,
                            do_align=False,
                            output_size=224,
                            margin=20,
                            max_frames=None,
                            target_fps=None,
                            skip_existing=False):
    """
    将视频处理成帧图片并保存

    参数：
    video_path: 视频文件路径
    output_dir: 输出目录
    frame_interval: 帧间隔（每N帧保存一帧，默认为1保存所有帧）
    do_align: 是否进行人脸对齐处理
    output_size: 人脸对齐后的输出大小
    margin: 人脸边界扩展像素
    max_frames: 最大处理帧数（None表示处理所有帧）
    target_fps: 目标帧率（如果指定，会按此帧率采样）
    skip_existing: 是否跳过已存在的帧文件

    返回：
    processed_frames: 处理的帧数
    """

    # 创建输出目录
    video_name = Path(video_path).stem
    frame_output_dir = Path(output_dir)

    # 初始化人脸检测器
    detector = MTCNN()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return 0

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"处理视频: {video_name}")
    print(f"  分辨率: {width}x{height}, FPS: {fps:.2f}, 总帧数: {total_frames}")
    print(f"  输出目录: {frame_output_dir}")
    print(f"  人脸对齐: {'是' if do_align else '否'}")

    # 计算实际帧间隔
    if target_fps and target_fps < fps:
        frame_interval = int(fps / target_fps)
        print(f"  目标FPS: {target_fps}, 帧间隔: {frame_interval}")

    processed_frames = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检查是否达到最大帧数限制
        if max_frames and processed_frames >= max_frames:
            break

        # 按帧间隔处理
        if frame_count % frame_interval == 0:
            # 生成文件名
            frame_filename = f"frame_{frame_count:06d}_{video_name}.jpg"
            frame_path = frame_output_dir / frame_filename

            # 如果跳过已存在的文件且文件已存在，则跳过
            if skip_existing and frame_path.exists():
                frame_count += 1
                continue

            processed_frame = frame.copy()

            # 如果需要人脸对齐
            if detector is not None:
                try:
                    # 转换颜色空间
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 检测人脸
                    results = detector.detect_faces(img_rgb)

                    if len(results) > 0:
                        result = results[0]
                        bounding_box = result['box']
                        landmarks = result['keypoints']

                        if do_align:
                            # 执行旋转对齐
                            left_eye = landmarks['left_eye']
                            right_eye = landmarks['right_eye']

                            # 计算眼睛中心点和角度
                            eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                                           (left_eye[1] + right_eye[1]) // 2)

                            dy = right_eye[1] - left_eye[1]
                            dx = right_eye[0] - left_eye[0]
                            angle = np.degrees(np.arctan2(dy, dx))

                            # 旋转矩阵
                            center = (eyes_center[0], eyes_center[1])
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)

                            # 执行旋转
                            rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                                                     flags=cv2.INTER_CUBIC)

                            # 计算旋转后的边界框
                            x, y, w, h = bounding_box
                            corners = np.array([
                                [x, y],
                                [x + w, y],
                                [x + w, y + h],
                                [x, y + h]
                            ])

                            ones = np.ones(shape=(len(corners), 1))
                            corners_homogeneous = np.hstack([corners, ones])
                            rotated_corners = M.dot(corners_homogeneous.T).T

                            # 计算旋转后的边界框
                            x_min = int(rotated_corners[:, 0].min() - margin)
                            x_max = int(rotated_corners[:, 0].max() + margin)
                            y_min = int(rotated_corners[:, 1].min() - margin)
                            y_max = int(rotated_corners[:, 1].max() + margin)

                            # 确保边界在图像范围内
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(rotated.shape[1], x_max)
                            y_max = min(rotated.shape[0], y_max)

                            # 裁剪人脸
                            processed_frame = rotated[y_min:y_max, x_min:x_max]
                        else:
                            # 不旋转，只裁剪
                            x, y, w, h = bounding_box
                            x_min = max(0, x - margin)
                            y_min = max(0, y - margin)
                            x_max = min(frame.shape[1], x + w + margin)
                            y_max = min(frame.shape[0], y + h + margin)
                            processed_frame = frame[y_min:y_max, x_min:x_max]

                        # 调整大小
                        if output_size:
                            # 计算缩放比例，保持宽高比不变
                            h, w = processed_frame.shape[:2]
                            scale = min(output_size / w, output_size / h)
                            new_w, new_h = int(w * scale), int(h * scale)

                            # 先按比例缩放到合适尺寸
                            resized_frame = cv2.resize(processed_frame, (new_w, new_h))

                            # 创建指定大小的画布，用黑色填充
                            canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)

                            # 计算居中位置
                            start_x = (output_size - new_w) // 2
                            start_y = (output_size - new_h) // 2

                            # 将缩放后的图像放置到画布中央
                            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_frame

                            processed_frame = canvas


                except Exception as e:
                    print(f"  帧 {frame_count} 人脸处理失败: {e}")

            # 保存帧
            cv2.imwrite(str(frame_path), processed_frame)
            processed_frames += 1

            # 显示进度
            if processed_frames % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  已处理 {processed_frames} 帧, 耗时: {elapsed:.1f}秒")

        frame_count += 1

    # 释放资源
    cap.release()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"  完成! 共保存 {processed_frames} 帧, 总耗时: {total_time:.1f}秒")
    if processed_frames > 0:
        print(f"  平均速度: {processed_frames / total_time:.1f} 帧/秒")

    return processed_frames


def process_video_directory(input_dir, output_dir=None,
                            video_extensions=['.mp4', '.avi', '.mov', '.mkv', '.flv'],
                            **kwargs):
    """
    处理目录中的所有视频文件

    参数：
    input_dir: 输入目录路径
    output_dir: 输出目录路径
    video_extensions: 视频文件扩展名列表
    **kwargs: 传递给process_video_to_frames的参数
    """

    input_path = Path(input_dir)
    # 如果没有指定输出目录，则在输入视频同目录下创建
    if output_dir is None:
        input_path = Path(input_dir)
        output_dir = input_path / "extracted_frames"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_path.glob(f"*{ext}")))


    print(f"在目录 {input_dir} 中找到 {len(video_files)} 个视频文件")

    total_frames = 0

    # 处理每个视频文件
    for video_file in video_files:
        print("\n" + "=" * 50)
        try:
            frames = process_video_to_frames(
                str(video_file),
                str(output_path),
                **kwargs
            )
            total_frames += frames
        except Exception as e:
            print(f"处理视频 {video_file.name} 时出错: {e}")

    print("\n" + "=" * 50)
    print(f"所有视频处理完成!")
    print(f"总处理帧数: {total_frames}")
    print(f"输出目录: {output_dir}")


# 使用示例
if __name__ == "__main__":
    # 方法1：直接调用函数处理整个目录
    process_video_directory(
        input_dir=r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total",
        # output_dir=r"D:\GraduationProject\demo1\output\frames",
        frame_interval=1,
        do_align=False,  # 进行人脸对齐
        output_size=224,
        margin=20,
        max_frames=None,  # 每个视频最多处理1000帧
        target_fps=None  # 目标帧率10fps

    )

    # 方法2：处理单个视频
    # process_video_to_frames(
    #     video_path="D:/GraduationProject/videos/sample.mp4",
    #     output_dir="D:/GraduationProject/frames",
    #     do_align=True
    # )
# 示例1：进行旋转对齐（默认）
#     aligned_face1, landmarks1 = align_face_mtcnn(r"D:\GraduationProject\demo1\output\frames\frame_000000_192.168.0.101_01_20231229150516_20231229151709.jpg",
#                                                  do_align=True
#                                                  )
#     if aligned_face1 is not None:
#         cv2.imwrite("aligned_rotated.jpg", aligned_face1)
#         print(f"旋转对齐模式：保存到 aligned_rotated.jpg")
#
#     # 示例2：不进行旋转对齐
#     aligned_face2, landmarks2 = align_face_mtcnn(r"D:\GraduationProject\demo1\output\frames\frame_000000_192.168.0.101_01_20231229150516_20231229151709.jpg",
#                                                  do_align=False
#                                                  )
#     if aligned_face2 is not None:
#         cv2.imwrite("aligned_cropped.jpg", aligned_face2)
#         print(f"简单裁剪模式：保存到 aligned_cropped.jpg")
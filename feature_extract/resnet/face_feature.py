from pathlib import Path
import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm


def letterbox_resize(img, target_size=(224, 224), color=(0, 0, 0)):
    """等比例缩放图片，多余部分用像素填充"""
    ih, iw = img.shape[:2]
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    dx, dy = (tw - nw) // 2, (th - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img_resized
    return canvas


def get_aligned_face(img, landmark):
    """根据双眼关键点旋转对齐人脸"""
    left_eye, right_eye = landmark[0], landmark[1]
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)


def process_single_image(img_path, output_full_path, mtcnn, is_align=True, margin=20, size=(224, 224)):
    """
    处理单张图片
    :param output_full_path: 完整的输出路径，例如 'D:/output/my_face.jpg'
    """
    img = cv2.imread(str(img_path))
    if img is None: return

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    if boxes is not None:
        # --- 情况1：检测到人脸 ---
        for i, (box, landmark) in enumerate(zip(boxes, landmarks)):

            target_img = get_aligned_face(img, landmark) if is_align else img
            x1, y1, x2, y2 = box
            nx1, ny1 = max(0, int(x1 - margin)), max(0, int(y1 - margin))
            nx2, ny2 = min(w, int(x2 + margin)), min(h, int(y2 + margin))

            face = target_img[ny1:ny2, nx1:nx2]
            if face.size > 0:
                face_final = letterbox_resize(face, target_size=size)
                cv2.imwrite(str(output_full_path), face_final)
                break  # 保存第一张高质量人脸后退出
    else:
        # --- 情况2：未检测到人脸 (新增逻辑) ---
        # 直接将原帧图像缩放到目标尺寸并保存
        # 注意：这里处理的是原图 img，而不是 img_rgb
        img_final = letterbox_resize(img, target_size=size)
        cv2.imwrite(str(output_full_path), img_final)


def process_single_video(video_path, output_dir, mtcnn, is_align=True, margin=20, size=(224, 224)):
    """处理单个视频并按指定格式保存文件名"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

        if boxes is not None and len(boxes) > 0:
            # 遍历所有人脸检测结果，找出置信度最高的
            best_face_idx = np.argmax(probs)
            best_box = boxes[best_face_idx]
            best_landmark = landmarks[best_face_idx]
            best_prob = probs[best_face_idx]

            if best_prob >= 0.95:
                # 高质量人脸：裁剪并保存人脸区域
                target_img = get_aligned_face(frame, best_landmark) if is_align else frame
                x1, y1, x2, y2 = best_box
                nx1, ny1 = max(0, int(x1 - margin)), max(0, int(y1 - margin))
                nx2, ny2 = min(w_vid, int(x2 + margin)), min(h_vid, int(y2 + margin))

                face = target_img[ny1:ny2, nx1:nx2]
                if face.size > 0:
                    face_final = letterbox_resize(face, target_size=size)
                else:
                    # 如果人脸区域无效，使用整帧
                    face_final = letterbox_resize(frame, target_size=size)
            else:
                # 置信度不够但有人脸检测：使用整帧
                face_final = letterbox_resize(frame, target_size=size)
        else:
            # 未检测到人脸：使用整帧
            face_final = letterbox_resize(frame, target_size=size)

        # 统一输出处理
        save_name = f"frame_{frame_idx:06d}_{video_path.stem}.jpg"
        cv2.imwrite(str(output_dir / save_name), face_final)

        frame_idx += 1
    cap.release()


def process_all_videos(input_dir, output_root, is_align=True, margin=20):
    """处理目录下所有视频"""
    input_path = Path(input_dir)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in input_path.iterdir() if f.suffix.lower() in video_exts]

    print(f"Device: {device} | Found {len(video_files)} videos.")

    for v_file in tqdm(video_files, desc="Processing Videos"):
        # 建议为每个视频创建一个子文件夹，或者直接混在一起保存
        # 如果你想混在一起保存，直接传 output_path
        # 如果想分文件夹，传 output_path / v_file.stem
        process_single_video(v_file, output_path, mtcnn, is_align, margin)


if __name__ == "__main__":
    # 配置你的路径
    INPUT_VIDEO_DIR = r'C:\Users\dengm\Desktop\dataset\blur_video'
    OUTPUT_BASE_DIR = r'C:\Users\dengm\Desktop\dataset\blur_video\extracted_frames'

    process_all_videos(
        input_dir=INPUT_VIDEO_DIR,
        output_root=OUTPUT_BASE_DIR,
        is_align=True,
        margin=30
    )

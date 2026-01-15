import threading
from datetime import datetime, timedelta  # 或者使用下面专门计时的 time 模块
import glob
import os
import re
import time  # 导入用于计时的标准 time 模块
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from pathlib import Path
from ultralytics import YOLO
import conf
from moviepy.editor import VideoFileClip
import multiprocessing

def find_main_person(
        image_path,
        model_weights=conf.model,
        center_weight=0.6,
        size_weight=0.4,
        conf_threshold=0.25,
        min_person_ratio=0.05,
        padding_ratio=0,
        return_type='cropped',
        use_person_class_only=True
):
    """
    在图片中找出最主要的人物（最靠近中心且面积最大），并进行裁剪。

    参数:
        image_path: 输入图片路径或numpy数组
        model_weights: YOLO模型权重路径，默认 'yolo11n.pt'
        center_weight: 中心位置权重 (0-1)，越高越偏好中心位置
        size_weight: 人物大小权重 (0-1)，越高越偏好大目标，两者之和应为1
        conf_threshold: 检测置信度阈值 (0-1)
        min_person_ratio: 最小人物面积占比，小于此值的目标将被忽略
        padding_ratio: 裁剪时额外添加的边距比例
        return_type: 返回类型:
            - 'cropped': 仅返回裁剪后的人物图像
            - 'annotated': 仅返回带标注的原图
            - 'all': 返回元组 (人物图像, 标注图像, 人物信息字典)
            - 'info': 仅返回人物信息字典
        use_person_class_only: 是否只检测 'person' 类别（True: 只识别人, False: 检测所有类别）

    返回:
        根据 return_type 返回相应结果，如果未检测到目标则返回 None
    """

    # 1. 加载图片
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path.copy()
    else:
        raise TypeError("image_path 必须是文件路径或numpy数组")

    img_height, img_width = img.shape[:2]
    image_center = (img_width / 2, img_height / 2)

    # 2. 加载模型
    try:
        model = YOLO(model_weights)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")

    # 3. 检测目标（根据参数决定是否只识别人）
    if use_person_class_only:
        results = model(img, classes=[0], conf=conf_threshold, verbose=False)
    else:
        results = model(img, conf=conf_threshold, verbose=False)
    # results[0].show()
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("未检测到目标")
        return None

    # 4. 处理检测结果，计算每个目标的得分
    persons_info = []
    image_area = img_width * img_height

    for i, box in enumerate(boxes):
        # 边界框坐标和类别
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = results[0].names[cls_id] if hasattr(results[0], 'names') else str(cls_id)

        # 计算目标面积和中心
        width = x2 - x1
        height = y2 - y1
        area = width * height
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 面积占比
        area_ratio = area / image_area

        # 跳过太小的目标
        # if area_ratio < min_person_ratio:
        #     continue

        # 计算中心距离比
        distance = np.sqrt((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2)
        max_distance = np.sqrt((img_width / 2) ** 2 + (img_height / 2) ** 2)
        distance_ratio = min(distance / max_distance, 1.0)

        # 综合得分
        score = (1 - distance_ratio) * center_weight + area_ratio * size_weight

        persons_info.append({
            'index': i,
            'bbox': (x1, y1, x2, y2),
            'bbox_int': (int(x1), int(y1), int(x2), int(y2)),
            'center': (center_x, center_y),
            'confidence': conf,
            'class_id': cls_id,
            'class_name': cls_name,
            'area': area,
            'area_ratio': area_ratio,
            'width': width,
            'height': height,
            'distance_ratio': distance_ratio,
            'score': score
        })

    if not persons_info:
        print(f"检测到目标但都小于最小尺寸要求 (min_person_ratio={min_person_ratio})")
        return None

    # 5. 找出得分最高的主要目标
    main_person = max(persons_info, key=lambda x: x['score'])

    print(f"共检测到 {len(persons_info)} 个有效目标")
    print(f"主要目标信息:")
    print(f"  - 类别: {main_person['class_name']} (ID: {main_person['class_id']})")
    print(f"  - 位置: {main_person['bbox_int']}")
    print(f"  - 置信度: {main_person['confidence']:.3f}")
    print(f"  - 面积占比: {main_person['area_ratio']:.3%}")
    print(f"  - 中心距离比: {main_person['distance_ratio']:.3f}")
    print(f"  - 综合得分: {main_person['score']:.3f}")

    # 6. 裁剪主要目标（添加边距）
    x1, y1, x2, y2 = main_person['bbox_int']
    person_width = x2 - x1
    person_height = y2 - y1

    # 计算边距
    pad_x = int(person_width * padding_ratio)
    pad_y = int(person_height * padding_ratio)

    # 应用边距，确保不超出图片边界
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(img_width, x2 + pad_x)
    y2_pad = min(img_height, y2 + pad_y)

    # 裁剪目标
    cropped_person = img[y1_pad:y2_pad, x1_pad:x2_pad]

    # 7. 创建标注图像（如果需要）
    if return_type in ['annotated', 'all']:
        annotated_img = img.copy()

        # 绘制所有检测到的目标
        for person in persons_info:
            px1, py1, px2, py2 = person['bbox_int']

            # 判断是否为当前主要目标
            if person['index'] == main_person['index']:
                color = (0, 0, 255)  # 红色：主要目标
                thickness = 3

                # 标注主要目标信息
                label = f"Main {person['class_name']}: {person['score']:.2f}"
                cv2.putText(annotated_img, label, (px1, py1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)

                # 绘制添加边距后的区域（黄色虚线）
                cv2.rectangle(annotated_img, (x1_pad, y1_pad),
                              (x2_pad, y2_pad), (0, 255, 255), 3, cv2.LINE_AA)
            else:
                # 根据类别分配颜色
                if person['class_id'] == 0:  # person
                    color = (0, 255, 0)  # 绿色
                else:
                    color = (255, 255, 0)  # 青色：其他类别
                thickness = 3

            # 绘制边界框
            cv2.rectangle(annotated_img, (px1, py1), (px2, py2), color, thickness)

            # 显示类别和置信度
            info_label = f"{person['class_name']}: {person['confidence']:.2f}"
            cv2.putText(annotated_img, info_label, (px1, py2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # 绘制图像中心点
        center_x, center_y = map(int, image_center)
        cv2.drawMarker(annotated_img, (center_x, center_y),
                       (255, 0, 0), cv2.MARKER_CROSS, 30, 3)

        # 在图像上显示统计信息
        stats_text = f"Objects: {len(persons_info)}  Main Score: {main_person['score']:.2f}"
        cv2.putText(annotated_img, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # 8. 根据返回类型返回结果
    if return_type == 'cropped':
        return cropped_person
    elif return_type == 'annotated':
        return annotated_img
    elif return_type == 'all':
        return cropped_person, annotated_img, main_person
    elif return_type == 'info':
        return main_person
    else:
        raise ValueError(f"无效的 return_type: {return_type}")


def find_main_person_and_inpaint(
        image_path,
        model_weights=conf.seg_model,
        center_weight=0.6,
        size_weight=0.4,
        conf_threshold=0.25,
        min_person_ratio=0.05,
        padding_ratio=0,
        inpaint_method='telea',  # 'telea' 或 'ns'
        inpaint_radius=3,
        return_type='processed',
        use_person_class_only=True
):
    """
    在图片中找出最主要的人物，并对其他人物区域进行修复填充，防止干扰。

    参数:
        image_path: 输入图片路径或numpy数组
        model_weights: YOLO分割模型权重路径，默认 'yolo11n-seg.pt'
        center_weight: 中心位置权重 (0-1)，越高越偏好中心位置
        size_weight: 人物大小权重 (0-1)，越高越偏好大目标，两者之和应为1
        conf_threshold: 检测置信度阈值 (0-1)
        min_person_ratio: 最小人物面积占比，小于此值的目标将被忽略
        padding_ratio: 裁剪时额外添加的边距比例
        inpaint_method: 修复算法，'telea' 或 'ns'
        inpaint_radius: 修复半径
        return_type: 返回类型:
            - 'processed': 仅返回处理后的图像（其他人填充）
            - 'cropped': 仅返回裁剪后主要人物图像
            - 'annotated': 仅返回带标注的原图
            - 'mask': 仅返回其他人物的掩码
            - 'all': 返回元组 (处理图像, 标注图像, 掩码, 人物信息)
        use_person_class_only: 是否只检测 'person' 类别

    返回:
        根据 return_type 返回相应结果，如果未检测到目标则返回 None
    """

    # 1. 加载图片
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path.copy()
    else:
        raise TypeError("image_path 必须是文件路径或numpy数组")

    img_height, img_width = img.shape[:2]
    image_center = (img_width / 2, img_height / 2)
    image_area = img_width * img_height

    # 2. 加载分割模型
    try:
        model = YOLO(model_weights)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")

    # 3. 运行分割检测
    if use_person_class_only:
        results = model(img, classes=[0], conf=conf_threshold, verbose=False)
    else:
        results = model(img, conf=conf_threshold, verbose=False)

    # 检查是否有分割结果
    if len(results) == 0 or results[0].masks is None:
        print("未检测到分割目标")
        return None

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("未检测到目标")
        return None

    # 4. 处理检测结果，计算每个目标的得分
    persons_info = []

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        # 边界框坐标和类别
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = results[0].names[cls_id] if hasattr(results[0], 'names') else str(cls_id)

        # 处理分割掩码
        mask_resized = cv2.resize(mask, (img_width, img_height))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        # 计算目标面积（基于分割掩码）
        mask_area = np.sum(mask_binary > 0)
        area_ratio = mask_area / image_area

        # 跳过太小的目标
        if area_ratio < min_person_ratio:
            continue

        # 计算中心位置（基于边界框）
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算中心距离比
        distance = np.sqrt((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2)
        max_distance = np.sqrt((img_width / 2) ** 2 + (img_height / 2) ** 2)
        distance_ratio = min(distance / max_distance, 1.0)

        # 综合得分
        score = (1 - distance_ratio) * center_weight + area_ratio * size_weight

        persons_info.append({
            'index': i,
            'bbox': (x1, y1, x2, y2),
            'bbox_int': (int(x1), int(y1), int(x2), int(y2)),
            'center': (center_x, center_y),
            'confidence': conf,
            'class_id': cls_id,
            'class_name': cls_name,
            'mask': mask_binary,
            'mask_area': mask_area,
            'area_ratio': area_ratio,
            'distance_ratio': distance_ratio,
            'score': score
        })

    if not persons_info:
        print(f"检测到目标但都小于最小尺寸要求 (min_person_ratio={min_person_ratio})")
        return None

    # 5. 找出得分最高的主要目标
    persons_info.sort(key=lambda x: x['score'], reverse=True)
    main_person = persons_info[0]
    other_persons = persons_info[1:]

    print(f"共检测到 {len(persons_info)} 个有效目标")
    print(f"主要目标信息:")
    print(f"  - 类别: {main_person['class_name']} (ID: {main_person['class_id']})")
    print(f"  - 位置: {main_person['bbox_int']}")
    print(f"  - 置信度: {main_person['confidence']:.3f}")
    print(f"  - 面积占比: {main_person['area_ratio']:.3%}")
    print(f"  - 中心距离比: {main_person['distance_ratio']:.3f}")
    print(f"  - 综合得分: {main_person['score']:.3f}")
    print(f"  - 需要修复的其他人物: {len(other_persons)} 个")

    # 6. 创建其他人物的掩码
    other_persons_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for person in other_persons:
        other_persons_mask = cv2.bitwise_or(other_persons_mask, person['mask'])

    # 7. 对其他人物的区域进行修复
    processed_img = img.copy()

    if len(other_persons) > 0 and np.sum(other_persons_mask) > 0:
        # 对掩码进行形态学操作，使边缘更平滑
        kernel = np.ones((5, 5), np.uint8)
        inpaint_mask = cv2.morphologyEx(other_persons_mask, cv2.MORPH_CLOSE, kernel)
        inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_OPEN, kernel)

        # 选择修复算法
        if inpaint_method == 'telea':
            processed_img = cv2.inpaint(processed_img, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
        elif inpaint_method == 'ns':
            processed_img = cv2.inpaint(processed_img, inpaint_mask, inpaint_radius, cv2.INPAINT_NS)
        else:
            raise ValueError(f"不支持的修复算法: {inpaint_method}")

        print(f"已修复 {np.sum(inpaint_mask > 0) / image_area:.2%} 的区域")

    # 8. 裁剪主要目标（如果需要）
    if return_type in ['cropped', 'all']:
        x1, y1, x2, y2 = main_person['bbox_int']
        person_width = x2 - x1
        person_height = y2 - y1

        # 计算边距
        pad_x = int(person_width * padding_ratio)
        pad_y = int(person_height * padding_ratio)

        # 应用边距，确保不超出图片边界
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(img_width, x2 + pad_x)
        y2_pad = min(img_height, y2 + pad_y)

        # 裁剪目标（从处理后的图像裁剪）
        cropped_person = processed_img[y1_pad:y2_pad, x1_pad:x2_pad]

    # 9. 创建标注图像（如果需要）
    if return_type in ['annotated', 'all']:
        annotated_img = img.copy()

        # 绘制所有检测到的目标
        for i, person in enumerate(persons_info):
            px1, py1, px2, py2 = person['bbox_int']

            # 判断是否为当前主要目标
            if i == 0:  # 主要目标
                color = (0, 0, 255)  # 红色：主要目标
                thickness = 5

                # 标注主要目标信息
                label = f"Main {person['class_name']}: {person['score']:.2f}"
                cv2.putText(annotated_img, label, (px1, py1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                # 绘制主要目标的掩码轮廓
                contours, _ = cv2.findContours(person['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_img, contours, -1, (0, 255, 255), 3)

                # 绘制添加边距后的区域（黄色虚线）
                if return_type in ['cropped', 'all']:
                    cv2.rectangle(annotated_img, (x1_pad, y1_pad),
                                  (x2_pad, y2_pad), (0, 255, 255), 3, cv2.LINE_AA)
            else:  # 其他人物（将被修复）
                color = (0, 255, 0)  # 绿色：将被修复的目标
                thickness = 3

                # 绘制修复区域的轮廓
                contours, _ = cv2.findContours(person['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_img, contours, -1, (255, 0, 0), 3)

                # 显示将被修复的标签
                label = f"Remove {person['class_name']}"
                cv2.putText(annotated_img, label, (px1, py2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # 绘制边界框
            cv2.rectangle(annotated_img, (px1, py1), (px2, py2), color, thickness)

        # 绘制图像中心点
        center_x, center_y = map(int, image_center)
        cv2.drawMarker(annotated_img, (center_x, center_y),
                       (255, 0, 0), cv2.MARKER_CROSS, 30, 3)

        # 在图像上显示统计信息
        stats_text = f"Total: {len(persons_info)}  Main Score: {main_person['score']:.2f}  To Remove: {len(other_persons)}"
        cv2.putText(annotated_img, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # 10. 根据返回类型返回结果
    if return_type == 'processed':
        return processed_img
    elif return_type == 'cropped':
        return cropped_person
    elif return_type == 'annotated':
        return annotated_img
    elif return_type == 'mask':
        return other_persons_mask
    elif return_type == 'all':
        return {
            'processed': processed_img,
            'annotated': annotated_img,
            'mask': other_persons_mask,
            'cropped': cropped_person,
            'main_person': main_person,
            'all_persons': persons_info
        }
    else:
        raise ValueError(f"无效的 return_type: {return_type}")


def find_main_face(
        image_path,
        model_weights=conf.face_model,
        center_weight=0.7,
        size_weight=0.3,
        conf_threshold=0.25,
        min_face_ratio=0.02,
        padding_ratio=0,
        return_type='cropped'
):
    """
    在图片中找出最主要的人脸（最靠近中心且面积最大），并进行裁剪。

    参数:
        image_path (str): 输入图片路径
        model_weights (str): YOLO人脸检测模型权重路径，默认 'yolov12n-face.pt'
        center_weight (float): 中心位置权重 (0-1)，越高越偏好中心位置
        size_weight (float): 人脸大小权重 (0-1)，越高越偏好大脸，两者之和应为1
        conf_threshold (float): 检测置信度阈值 (0-1)
        min_face_ratio (float): 最小人脸面积占比，小于此值的人脸将被忽略 (0-1)
        padding_ratio (float): 裁剪时额外添加的边距比例 (0-1)
        return_type (str): 返回类型，可选:
            - 'cropped': 仅返回裁剪后的人脸图像
            - 'annotated': 仅返回带标注的原图（标出所有人脸，主要人脸用红框）
            - 'all': 返回元组 (人脸图像, 标注图像, 人脸信息字典)
            - 'info': 仅返回人脸信息字典

    返回:
        根据 return_type 返回:
            - 裁剪的人脸图像 (numpy array)
            - 或标注图像 (numpy array)
            - 或元组 (人脸图像, 标注图像, 人脸信息字典)
            - 或人脸信息字典
        如果未检测到人脸，返回 None
    """

    # 1. 加载图片
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path.copy()
    else:
        raise TypeError("image_path 必须是文件路径或numpy数组")

    img_height, img_width = img.shape[:2]
    image_center = (img_width / 2, img_height / 2)

    # 2. 加载人脸检测模型
    try:
        model = YOLO(model_weights)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}\n请确保 '{model_weights}' 文件存在")

    # 3. 检测人脸
    results = model(img, conf=conf_threshold, verbose=False)
    # results[0].show()
    if len(results[0].boxes) == 0:
        print("未检测到人脸")
        return None

    # 4. 处理检测结果，计算每个脸的得分
    faces_info = []
    image_area = img_width * img_height

    for i, box in enumerate(results[0].boxes):
        # 边界框坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()

        # 计算人脸面积和中心
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2

        # 面积占比
        area_ratio = face_area / image_area

        # # 跳过太小的人脸
        # if area_ratio < min_face_ratio:
        #     continue

        # 计算中心距离比（归一化到0-1，0表示在中心）
        distance = np.sqrt((face_center_x - image_center[0]) ** 2 +
                           (face_center_y - image_center[1]) ** 2)
        max_distance = np.sqrt((img_width / 2) ** 2 + (img_height / 2) ** 2)
        distance_ratio = min(distance / max_distance, 1.0)

        # 综合得分：距离越近、面积越大，得分越高
        score = (1 - distance_ratio) * center_weight + area_ratio * size_weight

        faces_info.append({
            'index': i,
            'bbox': (x1, y1, x2, y2),
            'bbox_int': (int(x1), int(y1), int(x2), int(y2)),
            'center': (face_center_x, face_center_y),
            'confidence': conf,
            'area': face_area,
            'area_ratio': area_ratio,
            'distance_ratio': distance_ratio,
            'score': score
        })

    if not faces_info:
        print(f"检测到人脸但都小于最小尺寸要求 (min_face_ratio={min_face_ratio})")
        return None

    # 5. 找出得分最高的主要人脸
    main_face = max(faces_info, key=lambda x: x['score'])

    print(f"共检测到 {len(faces_info)} 张有效人脸")
    print(f"主要人脸信息:")
    print(f"  - 位置: [{main_face['bbox_int'][0]}, {main_face['bbox_int'][1]}, "
          f"{main_face['bbox_int'][2]}, {main_face['bbox_int'][3]}]")
    print(f"  - 置信度: {main_face['confidence']:.3f}")
    print(f"  - 面积占比: {main_face['area_ratio']:.3%}")
    print(f"  - 中心距离比: {main_face['distance_ratio']:.3f}")
    print(f"  - 综合得分: {main_face['score']:.3f}")

    # 6. 裁剪主要人脸（添加边距）
    x1, y1, x2, y2 = main_face['bbox_int']
    face_width = x2 - x1
    face_height = y2 - y1

    # 计算边距
    pad_x = int(face_width * padding_ratio)
    pad_y = int(face_height * padding_ratio)

    # 应用边距，确保不超出图片边界
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(img_width, x2 + pad_x)
    y2_pad = min(img_height, y2 + pad_y)

    # 裁剪人脸
    cropped_face = img[y1_pad:y2_pad, x1_pad:x2_pad]

    # 7. 创建标注图像（如果需要）
    if return_type in ['annotated', 'all']:
        annotated_img = img.copy()

        # 绘制所有人脸
        for face in faces_info:
            fx1, fy1, fx2, fy2 = face['bbox_int']

            # 判断是否为当前主要人脸
            if face['index'] == main_face['index']:
                color = (0, 0, 255)  # 红色：主要人脸
                thickness = 5

                # 标注主要人脸信息
                label = f"Main Face: {face['score']:.2f}"
                cv2.putText(annotated_img, label, (fx1, fy1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                # 绘制添加边距后的区域（黄色虚线）
                cv2.rectangle(annotated_img, (x1_pad, y1_pad),
                              (x2_pad, y2_pad), (0, 255, 255), 3, cv2.LINE_AA)
            else:
                color = (0, 255, 0)  # 绿色：其他人脸
                thickness = 3

            # 绘制人脸框
            cv2.rectangle(annotated_img, (fx1, fy1), (fx2, fy2), color, thickness)

            # 显示置信度
            conf_label = f"{face['confidence']:.2f}"
            cv2.putText(annotated_img, conf_label, (fx1, fy2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # 绘制图像中心点
        center_x, center_y = map(int, image_center)
        cv2.drawMarker(annotated_img, (center_x, center_y),
                       (255, 0, 0), cv2.MARKER_CROSS, 30, 3)

        # 在图像上显示统计信息
        stats_text = f"Faces: {len(faces_info)}  Main Score: {main_face['score']:.2f}"
        cv2.putText(annotated_img, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # 8. 根据返回类型返回结果
    if return_type == 'cropped':
        return cropped_face
    elif return_type == 'annotated':
        return annotated_img
    elif return_type == 'all':
        return cropped_face, annotated_img, main_face
    elif return_type == 'info':
        return main_face
    else:
        raise ValueError(f"无效的 return_type: {return_type}。可选: 'cropped', 'annotated', 'all', 'info'")


def find_main_person_and_blur(
        image_path,
        model_weights=conf.seg_model,
        center_weight=0.6,
        size_weight=0.4,
        conf_threshold=0.01,
        min_person_ratio=0.05,
        blur_strength=15,
        blur_method='gaussian',  # 'gaussian', 'median', 或 'bilateral'
        edge_smooth=True,
        edge_blur_radius=5,
        return_type='processed',
        use_person_class_only=True
):
    """
    在图片中找出最主要的人物，提取其精确轮廓，并将其他区域模糊处理。

    参数:
        image_path: 输入图片路径或numpy数组
        model_weights: YOLO分割模型权重路径，默认 'yolo11n-seg.pt'
        center_weight: 中心位置权重 (0-1)，越高越偏好中心位置
        size_weight: 人物大小权重 (0-1)，越高越偏好大目标，两者之和应为1
        conf_threshold: 检测置信度阈值 (0-1)
        min_person_ratio: 最小人物面积占比，小于此值的目标将被忽略
        blur_strength: 模糊强度（奇数）
        blur_method: 模糊方法:
            - 'gaussian': 高斯模糊
            - 'median': 中值模糊
            - 'bilateral': 双边滤波（保留边缘）
        edge_smooth: 是否平滑人物边缘
        edge_blur_radius: 边缘模糊半径，使过渡更自然
        return_type: 返回类型:
            - 'processed': 仅返回处理后的图像（其他人模糊）
            - 'mask': 仅返回主要人物掩码
            - 'blurred': 仅返回模糊的背景
            - 'alpha': 仅返回带透明度的主要人物
            - 'all': 返回所有结果
        use_person_class_only: 是否只检测 'person' 类别

    返回:
        根据 return_type 返回相应结果，如果未检测到目标则返回 None
    """

    # 1. 加载图片
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path.copy()
    else:
        raise TypeError("image_path 必须是文件路径或numpy数组")

    img_height, img_width = img.shape[:2]
    image_center = (img_width / 2, img_height / 2)
    image_area = img_width * img_height

    # 2. 加载分割模型
    try:
        model = YOLO(model_weights)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")

    # 3. 运行分割检测
    if use_person_class_only:
        results = model(img, classes=[0], conf=conf_threshold, verbose=False)
    else:
        results = model(img, conf=conf_threshold, verbose=False)

    # 检查是否有分割结果
    if len(results) == 0 or results[0].masks is None:
        print("未检测到分割目标")
        return None

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("未检测到目标")
        return None

    # 4. 处理检测结果，计算每个目标的得分
    persons_info = []

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        # 边界框坐标和类别
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = results[0].names[cls_id] if hasattr(results[0], 'names') else str(cls_id)

        # 处理分割掩码
        mask_resized = cv2.resize(mask, (img_width, img_height))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        # 计算目标面积（基于分割掩码）
        mask_area = np.sum(mask_binary > 0)
        area_ratio = mask_area / image_area

        # 跳过太小的目标
        if area_ratio < min_person_ratio:
            continue

        # 计算中心位置（基于边界框）
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算中心距离比
        distance = np.sqrt((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2)
        max_distance = np.sqrt((img_width / 2) ** 2 + (img_height / 2) ** 2)
        distance_ratio = min(distance / max_distance, 1.0)

        # 综合得分
        score = (1 - distance_ratio) * center_weight + area_ratio * size_weight

        persons_info.append({
            'index': i,
            'bbox': (x1, y1, x2, y2),
            'bbox_int': (int(x1), int(y1), int(x2), int(y2)),
            'center': (center_x, center_y),
            'confidence': conf,
            'class_id': cls_id,
            'class_name': cls_name,
            'mask': mask_binary,
            'mask_area': mask_area,
            'area_ratio': area_ratio,
            'distance_ratio': distance_ratio,
            'score': score
        })

    if not persons_info:
        print(f"检测到目标但都小于最小尺寸要求 (min_person_ratio={min_person_ratio})")
        return None

    # 5. 找出得分最高的主要目标
    persons_info.sort(key=lambda x: x['score'], reverse=True)
    main_person = persons_info[0]
    other_persons = persons_info[1:]

    # print(f"共检测到 {len(persons_info)} 个有效目标")
    # print(f"主要目标信息:")
    # print(f"  - 类别: {main_person['class_name']} (ID: {main_person['class_id']})")
    # print(f"  - 位置: {main_person['bbox_int']}")
    # print(f"  - 置信度: {main_person['confidence']:.3f}")
    # print(f"  - 面积占比: {main_person['area_ratio']:.3%}")
    # print(f"  - 中心距离比: {main_person['distance_ratio']:.3f}")
    # print(f"  - 综合得分: {main_person['score']:.3f}")
    # print(f"  - 需要模糊的其他人物: {len(other_persons)} 个")

    # 6. 创建主要人物的精确掩码
    main_person_mask = main_person['mask']

    # 对主要人物掩码进行后处理，使其更平滑
    if edge_smooth:
        # 形态学操作平滑边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        main_person_mask = cv2.morphologyEx(main_person_mask, cv2.MORPH_CLOSE, kernel)
        main_person_mask = cv2.morphologyEx(main_person_mask, cv2.MORPH_OPEN, kernel)

        # 高斯模糊边缘创建柔和过渡
        main_person_mask_float = main_person_mask.astype(np.float32) / 255.0
        main_person_mask_blurred = cv2.GaussianBlur(main_person_mask_float,
                                                    (edge_blur_radius * 2 + 1, edge_blur_radius * 2 + 1),
                                                    0)
        main_person_mask = (main_person_mask_blurred * 255).astype(np.uint8)

    # 7. 创建其他人物的掩码
    other_persons_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for person in other_persons:
        other_persons_mask = cv2.bitwise_or(other_persons_mask, person['mask'])

    # 8. 创建背景区域掩码（需要模糊的区域）
    background_mask = cv2.bitwise_not(main_person_mask)
    # 确保其他人物的区域也被包含在背景中
    background_mask = cv2.bitwise_or(background_mask, other_persons_mask)

    # 9. 对背景区域进行模糊处理
    blurred_background = img.copy()

    # 确保模糊强度是奇数
    if blur_strength % 2 == 0:
        blur_strength += 1

    # 选择模糊方法
    if blur_method == 'gaussian':
        blurred_background = cv2.GaussianBlur(blurred_background,
                                              (blur_strength, blur_strength), 0)
    elif blur_method == 'median':
        blurred_background = cv2.medianBlur(blurred_background, blur_strength)
    elif blur_method == 'bilateral':
        blurred_background = cv2.bilateralFilter(blurred_background,
                                                 blur_strength,
                                                 75, 75)
    else:
        raise ValueError(f"不支持的模糊方法: {blur_method}")

    # 10. 创建最终图像：主要人物保持清晰，背景模糊
    # 创建alpha混合掩码
    if edge_smooth:
        alpha_mask = main_person_mask_blurred
    else:
        alpha_mask = main_person_mask.astype(np.float32) / 255.0

    # 扩展alpha掩码到3个通道
    alpha_mask_3channel = np.stack([alpha_mask] * 3, axis=2)

    # 混合清晰的主要人物和模糊的背景
    final_image = (img.astype(np.float32) * alpha_mask_3channel +
                   blurred_background.astype(np.float32) * (1 - alpha_mask_3channel))
    final_image = final_image.astype(np.uint8)

    if return_type in ['annotated', 'all']:
        annotated_img = img.copy()

        # 绘制所有检测到的目标
        for i, person in enumerate(persons_info):
            px1, py1, px2, py2 = person['bbox_int']

            # 判断是否为当前主要目标
            if i == 0:  # 主要目标
                color = (0, 0, 255)  # 红色：主要目标
                thickness = 5

                # 标注主要目标信息
                label = f"Main {person['class_name']}: {person['score']:.2f}"
                cv2.putText(annotated_img, label, (px1, py1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                # 绘制主要目标的掩码轮廓
                contours, _ = cv2.findContours(person['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_img, contours, -1, (0, 255, 255), 3)

            else:  # 其他人物（将被模糊）
                color = (0, 255, 0)  # 绿色：将被模糊的目标
                thickness = 3

                # 绘制模糊区域的轮廓
                contours, _ = cv2.findContours(person['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_img, contours, -1, (255, 0, 0), 3)

                # 显示将被模糊的标签
                label = f"Remove {person['class_name']}"
                cv2.putText(annotated_img, label, (px1, py2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # 绘制边界框
            cv2.rectangle(annotated_img, (px1, py1), (px2, py2), color, thickness)

        # 绘制图像中心点
        center_x, center_y = map(int, image_center)
        cv2.drawMarker(annotated_img, (center_x, center_y),
                       (255, 0, 0), cv2.MARKER_CROSS, 30, 3)

        # 在图像上显示统计信息
        stats_text = f"Total: {len(persons_info)}  Main Score: {main_person['score']:.2f}  To Remove: {len(other_persons)}"
        cv2.putText(annotated_img, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    # 11. 创建透明的主要人物图像（RGBA格式）
    if return_type in ['alpha', 'all']:
        # 创建RGBA图像
        alpha_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # 设置alpha通道
        if edge_smooth:
            alpha_channel = (main_person_mask_blurred * 255).astype(np.uint8)
        else:
            alpha_channel = main_person_mask

        # 应用alpha通道
        alpha_image[:, :, 3] = alpha_channel

        # 可选：将背景设为完全透明
        alpha_image[alpha_channel == 0] = [0, 0, 0, 0]

    # 12. 创建可视化掩码（用于调试）
    if return_type in ['all']:
        # 创建彩色掩码可视化
        mask_visual = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        mask_visual[main_person_mask > 0] = [0, 255, 0]  # 绿色：主要人物
        mask_visual[other_persons_mask > 0] = [255, 0, 0]  # 红色：其他人

    # 13. 根据返回类型返回结果
    if return_type == 'processed':
        return final_image
    elif return_type == 'mask':
        return main_person_mask
    elif return_type == 'blurred':
        return blurred_background
    elif return_type == 'alpha':
        return alpha_image
    elif return_type == 'annotated':
        return annotated_img
    elif return_type == 'all':
        return {
            'processed': final_image,
            'annotated': annotated_img,
            'main_mask': main_person_mask,
            'alpha_person': alpha_image,
            'blurred_bg': blurred_background,
            'mask_visual': mask_visual,
            'main_person': main_person,
            'all_persons': persons_info,
            'original': img
        }
    else:
        raise ValueError(f"无效的 return_type: {return_type}")


def process_video_with_blur(
        video_path,
        output_path=None,
        frame_interval=1,  # 处理间隔，1表示处理每一帧
        target_fps=None,  # 输出视频的FPS，None表示使用原视频FPS
        target_resolution=None,  # 输出视频分辨率 (width, height)，None表示使用原分辨率
        show_preview=False,  # 是否显示实时预览
        progress_bar=True,  # 是否显示进度条
        save_frames=False,  # 是否保存每一帧为图片
        frames_output_dir=None,  # 帧保存目录
        **blur_kwargs  # 传递给find_main_person_and_blur的参数
):
    """
    处理视频，对每一帧应用find_main_person_and_blur函数

    参数:
        video_path: 输入视频路径
        output_path: 输出视频路径，如果为None则不保存视频
        frame_interval: 帧处理间隔，1=每帧，2=每2帧处理1帧，等等
        target_fps: 输出视频的帧率，None表示使用原视频帧率
        target_resolution: 输出视频分辨率 (width, height)，None表示使用原分辨率
        show_preview: 是否显示实时预览窗口
        progress_bar: 是否显示处理进度条
        save_frames: 是否保存每一帧为图片
        frames_output_dir: 帧保存目录
        **blur_kwargs: 传递给find_main_person_and_blur的参数

    返回:
        处理后的视频统计信息
    """

    # 1. 打开视频文件
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    # 2. 获取视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用目标FPS或原始FPS
    output_fps = target_fps if target_fps is not None else original_fps
    output_fps = max(1, output_fps)  # 确保FPS至少为1

    # 使用目标分辨率或原始分辨率
    if target_resolution is not None:
        output_width, output_height = target_resolution
    else:
        output_width, output_height = original_width, original_height

    print(f"视频信息:")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 原始FPS: {original_fps:.2f}")
    print(f"  - 原始分辨率: {original_width}x{original_height}")
    print(f"  - 输出FPS: {output_fps:.2f}")
    print(f"  - 输出分辨率: {output_width}x{output_height}")
    print(f"  - 帧处理间隔: {frame_interval}")

    # 3. 准备输出视频写入器
    output_video_writer = None
    if output_path is not None:
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 根据文件扩展名选择编码器
        output_ext = Path(output_path).suffix.lower()

        if output_ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        elif output_ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI常用编码器

        elif output_ext == '.mov':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MOV格式

        elif output_ext == '.mkv':
            fourcc = cv2.VideoWriter_fourcc(*'X264')  # MKV常用编码器

        elif output_ext == '.flv':
            fourcc = cv2.VideoWriter_fourcc(*'FLV1')  # FLV格式

        elif output_ext == '.wmv':
            fourcc = cv2.VideoWriter_fourcc(*'WMV2')  # WMV格式

        else:
            # 默认使用MP4V编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            print(f"  - 警告: 未知扩展名 {output_ext}，使用默认编码器 'mp4v'")

        print(f"  - 自动选择编码器: {fourcc} 用于扩展名 {output_ext}")

        output_video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            output_fps,
            (output_width, output_height)
        )

        if not output_video_writer.isOpened():
            print(f"警告: 无法创建输出视频文件: {output_path}")
            output_video_writer = None

    # 4. 准备帧保存目录（如果需要）
    if save_frames and frames_output_dir is not None:
        frames_dir = Path(frames_output_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"帧将保存到: {frames_dir}")

    # 5. 准备进度条
    if progress_bar:
        try:
            from tqdm import tqdm
            progress = tqdm(total=total_frames // frame_interval,
                            desc="处理视频", unit="帧")
        except ImportError:
            print("提示: 安装tqdm可以获得进度条: pip install tqdm")
            progress_bar = False

    # 6. 处理视频帧
    processed_frames = 0
    skipped_frames = 0
    total_processing_time = 0

    frame_index = 0
    output_frame_index = 0

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 跳过不需要处理的帧
        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        # 开始处理时间
        start_time = time.time()

        try:
            # 应用find_main_person_and_blur函数
            processed_frame = find_main_person_and_blur(
                image_path=frame,
                return_type='processed',  # 只返回处理后的图像
                **blur_kwargs
            )

            # 如果处理失败（返回None），使用原始帧
            if processed_frame is None:
                processed_frame = frame
                print(f"帧 {frame_index}: 未检测到人物，使用原始帧")

            # 调整分辨率（如果需要）
            if (processed_frame.shape[1], processed_frame.shape[0]) != (output_width, output_height):
                processed_frame = cv2.resize(processed_frame, (output_width, output_height))

            # 保存处理后的帧到视频文件
            if output_video_writer is not None:
                output_video_writer.write(processed_frame)

            # 保存帧为图片（如果需要）
            if save_frames and frames_output_dir is not None:
                frame_filename = f"frame_{output_frame_index:06d}.jpg"
                frame_path = Path(frames_output_dir) / frame_filename
                cv2.imwrite(str(frame_path), processed_frame)

            # 显示实时预览（如果需要）
            if show_preview:
                # 创建预览窗口
                preview_frame = processed_frame.copy()

                # 如果分辨率太大，缩小显示
                max_preview_size = 800
                if preview_frame.shape[1] > max_preview_size:
                    scale = max_preview_size / preview_frame.shape[1]
                    new_width = max_preview_size
                    new_height = int(preview_frame.shape[0] * scale)
                    preview_frame = cv2.resize(preview_frame, (new_width, new_height))

                # 添加帧信息
                info_text = f"Frame: {frame_index}/{total_frames}  Processed: {output_frame_index}"
                cv2.putText(preview_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                fps_text = f"FPS: {1 / (time.time() - start_time + 0.001):.1f}"
                cv2.putText(preview_frame, fps_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Video Processing Preview', preview_frame)

                # 按'q'键退出预览
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断处理")
                    break

            processed_frames += 1
            output_frame_index += 1

        except Exception as e:
            print(f"帧 {frame_index} 处理失败: {e}")
            skipped_frames += 1
            # 处理失败时写入原始帧
            if output_video_writer is not None:
                resized_frame = cv2.resize(frame, (output_width, output_height))
                output_video_writer.write(resized_frame)

        # 计算处理时间
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        # 更新进度条
        if progress_bar:
            progress.update(1)
            progress.set_postfix({
                'fps': f"{1 / processing_time:.1f}" if processing_time > 0 else "0",
                'time': f"{processing_time:.3f}s"
            })

        frame_index += 1

    # 7. 清理资源
    if progress_bar:
        progress.close()

    cap.release()
    if output_video_writer is not None:
        output_video_writer.release()

    if show_preview:
        cv2.destroyAllWindows()

    # 8. 计算统计信息
    avg_processing_time = total_processing_time / max(processed_frames, 1)
    estimated_total_time = total_frames * avg_processing_time / frame_interval

    print(f"\n视频处理完成!")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 处理帧数: {processed_frames}")
    print(f"  - 跳过帧数: {skipped_frames}")
    print(f"  - 平均处理时间: {avg_processing_time:.3f}秒/帧")
    print(f"  - 总处理时间: {total_processing_time:.2f}秒")
    print(f"  - 预估全处理时间: {estimated_total_time:.2f}秒")

    if output_video_writer is not None:
        print(f"  - 输出视频已保存到: {output_path}")

    if save_frames and frames_output_dir is not None:
        print(f"  - 帧图片已保存到: {frames_output_dir}")

    return {
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'skipped_frames': skipped_frames,
        'avg_processing_time': avg_processing_time,
        'total_processing_time': total_processing_time,
        'output_path': output_path,
        'output_resolution': (output_width, output_height),
        'output_fps': output_fps
    }


def process_video(input_video, output_video):
    """
    处理视频的简单示例
    """
    input_video = input_video
    output_video = output_video

    # 基本参数
    blur_params = {
        'model_weights': conf.seg_model,
        'center_weight': 0.6,
        'size_weight': 0.4,
        'conf_threshold': 0.25,
        'min_person_ratio': 0,
        'blur_strength': 200,
        'blur_method': 'gaussian',
        'edge_smooth': True,
        'edge_blur_radius': 7,
        'use_person_class_only': True
    }

    # 处理视频
    result = process_video_with_blur(
        video_path=input_video,
        output_path=output_video,
        frame_interval=1,  # 处理每一帧
        # target_fps=30,  # 输出30fps
        show_preview=True,  # 显示预览
        progress_bar=True,  # 显示进度条
        **blur_params
    )

    return result


def batch_process_videos(input_dir, output_dir=None, **blur_params):
    """
    批量处理目录中的所有视频文件

    Args:
        input_dir: 输入视频目录路径
        output_dir: 输出目录路径（可选，默认在输入目录下创建'processed'文件夹）
        **blur_params: 处理参数，与process_video_with_blur的参数一致
    """

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, "processed")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']

    # 获取所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        # video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    if not video_files:
        print(f"在目录 {input_dir} 中未找到视频文件")
        print(f"支持格式: {', '.join(video_extensions)}")
        return []

    print(f"找到 {len(video_files)} 个视频文件:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video)}")
    print()

    # 默认处理参数（可根据需要修改）
    default_params = {
        'model_weights': conf.seg_model,
        'center_weight': 0.6,
        'size_weight': 0.4,
        'conf_threshold': 0.25,
        'min_person_ratio': 0,
        'blur_strength': 200,
        'blur_method': 'gaussian',
        'edge_smooth': True,
        'edge_blur_radius': 7,
        'use_person_class_only': True
    }

    # 更新默认参数
    default_params.update(blur_params)
    # 初始化计数器
    completed_count = 0
    # 多线程处理
    max_workers = min(len(video_files), multiprocessing.cpu_count())
    # 处理结果列表
    results = []

    # 批量处理

    total_files = len(video_files)


    # for i, video_path in enumerate(video_files, 1):
    #     try:
    #         print(f"\n{'=' * 60}")
    #         print(f"正在处理文件 {i}/{total_files}: {os.path.basename(video_path)}")
    #         print(f"{'=' * 60}")
    #
    #         # 生成输出文件路径
    #         video_name = Path(video_path).stem
    #         output_filename = f"{video_name}_blurred.mp4"
    #         output_path = os.path.join(output_dir, output_filename)
    #
    #         # 如果文件已存在，添加时间戳
    #         if os.path.exists(output_path):
    #             timestamp = time.strftime("%Y%m%d_%H%M%S")
    #             output_filename = f"{video_name}_blurred_{timestamp}.mp4"
    #             output_path = os.path.join(output_dir, output_filename)
    #
    #         # 记录开始时间
    #         file_start_time = time.time()
    #
    #         # 处理单个视频
    #         result = process_video_with_blur(
    #             video_path=video_path,
    #             output_path=output_path,
    #             **default_params
    #         )
    #
    #         # 计算处理时间
    #         file_time = time.time() - file_start_time
    #
    #         # 保存处理结果
    #         result_info = {
    #             'input_file': video_path,
    #             'output_file': output_path,
    #             'success': True,
    #             'processing_time': file_time,
    #             'details': result
    #         }
    #         results.append(result_info)
    #
    #         print(f"✓ 处理完成: {os.path.basename(output_path)}")
    #         print(f"  处理时间: {file_time:.2f}秒")
    #
    #
    #
    #     except Exception as e:
    #         print(f"✗ 处理失败: {str(e)}")
    #         error_info = {
    #             'input_file': video_path,
    #             'error': str(e),
    #             'success': False
    #         }
    #         results.append(error_info)
    # 使用线程锁保护共享资源
    lock = threading.Lock()

    def process_single_video(video_path):
        """处理单个视频的函数"""
        nonlocal completed_count
        try:
            # 生成输出文件路径
            video_name = Path(video_path).stem
            output_filename = f"{video_name}_blurred.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # 如果文件已存在，添加时间戳
            if os.path.exists(output_path):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"{video_name}_blurred_{timestamp}.mp4"
                output_path = os.path.join(output_dir, output_filename)

            # 记录开始时间
            file_start_time = time.time()

            # 调用单个视频处理函数
            result = process_video_with_blur(
                video_path=video_path,
                output_path=output_path,
                **default_params
            )

            # 计算处理时间
            file_time = time.time() - file_start_time

            # 保存处理结果
            result_info = {
                'input_file': video_path,
                'output_file': output_path,
                'success': True,
                'processing_time': file_time,
                'details': result
            }

            # 打印完成信息（线程安全）
            with lock:

                completed_count += 1
                print(f"[{completed_count}/{total_files}] ✓ 处理完成: {os.path.basename(output_path)} "
                      f"(耗时: {file_time:.2f}秒)")

            return result_info

        except Exception as e:
            # 处理错误（线程安全）
            with lock:

                completed_count += 1
                print(f"[{completed_count}/{total_files}] ✗ 处理失败: {str(e)}")

            error_info = {
                'input_file': video_path,
                'error': str(e),
                'success': False
            }
            return error_info

        # 使用线程池并行处理

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_video = {executor.submit(process_single_video, video): video
                           for video in video_files}

        # 收集结果
        for future in as_completed(future_to_video):
            result = future.result()
            results.append(result)

    # 汇总统计
    total_time = time.time() - start_time

    successful = sum(1 for r in results if r.get('success', False))
    failed = total_files - successful

    print(f"\n{'=' * 60}")
    print("批量处理完成！")
    print(f"{'=' * 60}")
    print(f"总计处理: {total_files} 个文件")
    print(f"成功: {successful} 个")
    print(f"失败: {failed} 个")
    print(f"总耗时: {time.strftime('%H小时%M分钟%S秒', time.gmtime(total_time))}秒")
    print(f"平均每个文件: {time.strftime('%H小时%M分钟%S秒', time.gmtime(total_time / total_files)):.2f}秒")
    print(f"输出目录: {output_dir}")

    return results


def cut_video_from_time_to_end(input_video_path, target_start_time, target_end_time=None, output_dir="output"):
    """
    根据绝对时间剪辑视频，从目标时间开始直到视频结束

    参数:
    - input_video_path: 输入视频文件路径
    - target_time_str: 目标开始时间字符串，纯数字格式："20231229153000"
    - output_dir: 输出目录
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 解析输入视频文件名
    filename = os.path.basename(input_video_path)

    try:
        # 提取视频开始时间
        # 使用正则表达式查找时间戳
        match = re.search(r'_(\d{14})_', filename)
        if match:
            time_str = match.group(1)

        else:
            raise ValueError(f"无法从文件名 {filename} 中提取时间信息")
        video_start_time = datetime.strptime(time_str, '%Y%m%d%H%M%S')
        print(f"视频开始时间: {video_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 解析目标时间
        target_start_time = datetime.strptime(target_start_time, '%Y%m%d%H%M%S')
        print(f"目标开始时间: {target_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 计算时间差（秒）
        time_diff = (target_start_time - video_start_time).total_seconds()

        if time_diff < 0:
            print(
                f"警告：目标时间 {target_start_time.strftime('%Y-%m-%d %H:%M:%S')} 早于视频开始时间 {video_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return
        else:
            start_seconds = time_diff

        # 加载视频文件
        print(f"加载视频: {input_video_path}")
        video = VideoFileClip(input_video_path)

        # 获取视频总时长
        video_duration = video.duration
        video_end_time = video_start_time + timedelta(seconds=video_duration)
        print(f"视频总时长: {video_duration:.2f}秒\n"
              f"目标开始于 {target_start_time.strftime('%Y-%m-%d %H:%M:%S')}结束于{video_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        # 检查开始时间是否超过视频时长
        if start_seconds >= video_duration:
            raise ValueError(f"目标开始时间超出视频时长（视频时长: {video_duration:.2f}秒）")

        # 计算实际剪辑时长
        # 确定结束时间
        if target_end_time is None:
            # 默认截取到视频结束
            end_seconds = video_duration

            print("未指定结束时间，将截取到视频结束")
        else:
            # 解析指定的结束时间
            video_end_time = datetime.strptime(target_end_time, '%Y%m%d%H%M%S')
            print(f"目标结束时间: {video_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 计算结束时间差（秒）
            end_time_diff = (video_end_time - video_start_time).total_seconds()
            end_seconds = end_time_diff

            # 验证结束时间是否合理
            if end_seconds <= start_seconds:
                raise ValueError(f"结束时间必须晚于开始时间")

            if end_seconds > video_duration:
                print(f"警告：指定的结束时间晚于视频结束时间，将截取到视频结束")
                end_seconds = video_duration

        # 剪辑视频（从目标时间开始）
        subclip = video.subclip(start_seconds, end_seconds)

        # 生成输出文件名
        # 使用纯数字时间格式作为文件名
        output_filename = f"{filename.rsplit('_', 2)[0]}_{target_start_time.strftime('%Y%m%d%H%M%S')}_{video_end_time.strftime('%Y%m%d%H%M%S')}.MP4"
        output_path = os.path.join(output_dir, output_filename)

        # 保存剪辑后的视频
        print(f"正在保存剪辑视频到: {output_path}")
        subclip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec=False,
            verbose=False,  # 减少输出信息
            logger=None,  # 关闭logger，减少输出
            threads=12  # 使用多线程加速处理
        )

        # 清理资源
        video.close()
        subclip.close()

        print(f"✓ 视频剪辑完成: {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ 处理视频时发生错误: {str(e)}")
        raise


def get_video_end_time(video_path):
    """
    获取视频的结束时间

    参数:
    - video_path: 视频文件路径

    返回:
    - 视频的结束时间
    """

    # 解析输入视频文件名
    filename = os.path.basename(video_path)

    try:
        # 提取视频开始时间
        # 使用正则表达式查找时间戳
        match = re.search(r'_(\d{14})_', filename)
        if match:
            time_str = match.group(1)

        else:
            raise ValueError(f"无法从文件名 {filename} 中提取时间信息")
        video_start_time = datetime.strptime(time_str, '%Y%m%d%H%M%S')
        print(f"视频开始时间: {video_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        video = VideoFileClip(video_path)

        # 获取视频总时长
        video_duration = video.duration
        video_end_time = video_start_time + timedelta(seconds=video_duration)
        print(f"视频总时长: {video_duration:.2f}秒\n"
              f"开始于 {video_start_time.strftime('%Y-%m-%d %H:%M:%S')}结束于{video_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        output_filename = f"{filename.rsplit('_', 2)[0]}_{video_start_time.strftime('%Y%m%d%H%M%S')}_{video_end_time.strftime('%Y%m%d%H%M%S')}.MP4"
        dir_path = os.path.dirname(video_path)
        output_path = os.path.join(dir_path, output_filename)
        os.rename(video_path, output_path)
    except Exception as e:
        print(f"✗ 获取视频结束时间发生错误: {str(e)}")
        raise


def batch_rename_videos_with_duration(directory_path):
    """
    遍历目录中的视频文件，根据时长重命名文件

    参数:
    - directory_path: 视频所在的目录路径
    """

    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"✗ 错误：目录不存在 - {directory_path}")
        return

    print(f"正在扫描目录: {directory_path}")

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # 跳过目录，只处理文件
        if not os.path.isfile(file_path):
            continue

        # 只处理 .mp4 文件 (忽略大小写)
        if not filename.lower().endswith('.mp4'):
            # 如果日志文件或其他文件，跳过
            continue

        try:
            # 1. 提取视频开始时间 (使用正则匹配)
            match = re.search(r'_(\d{14})_', filename)
            if not match:
                print(f"⚠ 跳过 {filename}：无法从文件名中提取时间信息 (需要14位数字)")
                continue

            time_str = match.group(1)
            video_start_time = datetime.strptime(time_str, '%Y%m%d%H%M%S')

            # 2. 获取视频时长
            # 使用 with 语句，确保处理完后释放文件句柄，否则 os.rename 会报错
            with VideoFileClip(file_path) as video:
                video_duration = video.duration

            # 3. 计算结束时间
            video_end_time = video_start_time + timedelta(seconds=video_duration)

            # 4. 构造新文件名
            # 原逻辑是去掉后面，这里我们改为更稳健的逻辑：
            # 新格式: 原文件名_开始时间_结束时间.MP4
            # 格式化时间
            start_str = video_start_time.strftime('%Y%m%d%H%M%S')
            end_str = video_end_time.strftime('%Y%m%d%H%M%S')

            # 拼接新文件名：原文件名_开始时间_结束时间
            new_filename = f"{filename.rsplit('_', 2)[0]}_{start_str}_{end_str}.MP4"
            new_path = os.path.join(directory_path, new_filename)

            # 5. 重命名
            # 如果新旧文件名一样，跳过 (防止重复处理)
            if filename == new_filename:
                continue

            os.rename(file_path, new_path)

            print(f"✅ 成功: {filename}")
            print(f"   时长: {video_duration:.2f}秒")
            print(f"   重命名为: {new_filename}\n")

        except Exception as e:
            print(f"✗ 处理失败 [{filename}]: {str(e)}\n")


if __name__ == '__main__':
    blur_params = {
        'model_weights': conf.seg_model,
        'center_weight': 0.6,
        'size_weight': 0.4,
        'conf_threshold': 0.25,
        'min_person_ratio': 0,
        'blur_strength': 200,
        'blur_method': 'gaussian',
        'edge_smooth': True,
        'edge_blur_radius': 7,
        'use_person_class_only': True
    }
    # batch_process_videos(input_dir=r"E:\数据\20231229 计算机网络考试数据汇总\第2组\视频\2021214372_姬高阳",
    #                      **blur_params)

    # cut_video_from_time_to_end(
    #     r"E:\数据\20231229 计算机网络考试数据汇总\第6组\视频\2021214398_张颖\total\192.168.0.124_01_20231229160026_blurred.mp4",
    #     "20231229160026",
    #     "20231229160416",
    #     r"E:\数据\20231229 计算机网络考试数据汇总\第6组\视频\2021214398_张颖\total\新建文件夹")
    # get_video_end_time(
    #     r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\新建文件夹\192.168.0.101_01_20231229150209_blurred.mp4")
    batch_rename_videos_with_duration(r"E:\数据\20231229 计算机网络考试数据汇总\第6组\视频\2021214398_张颖\total\新建文件夹")
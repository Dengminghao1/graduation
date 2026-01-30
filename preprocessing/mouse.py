import cv2
import numpy as np
import time
import math
from pathlib import Path
import json
import csv
from datetime import datetime


# 全局变量，用于存储前一帧的灰度图（帧差距分析）
_prev_frame_gray = None

def detect_mouse_pointer(frame, threshold=0.5, min_size=50, max_size=500, template=None):
    """
    在视频帧中检测鼠标指针
    
    参数:
        frame: 输入视频帧 (numpy数组)
        threshold: 阈值，用于二值化处理 (0-1)
        min_size: 鼠标指针最小大小
        max_size: 鼠标指针最大大小
        template: 鼠标模板图像 (numpy数组或列表)，如果提供则使用模板匹配
    
    返回:
        tuple: (x, y, radius) 鼠标指针位置和大小，如果未检测到返回 None
    """
    global _prev_frame_gray
    
    # 如果提供了模板，使用模板匹配
    if template is not None:
        # 处理模板列表（多张模板）
        if isinstance(template, list):
            for temp in template:
                result = match_mouse_template(frame, temp)
                if result:
                    return result
        else:
            # 单张模板
            result = match_mouse_template(frame, template)
            if result:
                return result
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 方法0: 使用帧差距分析（针对鼠标移动）
    if _prev_frame_gray is not None:
        # 计算当前帧与前一帧的差异
        frame_diff = cv2.absdiff(blurred, _prev_frame_gray)
        
        # 对差异图像进行阈值处理
        _, diff_binary = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        diff_contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓，寻找可能的鼠标移动
        for contour in diff_contours:
            area = cv2.contourArea(contour)
            
            # 过滤大小（鼠标移动通常是小区域）
            if min_size < area < max_size:
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算中心点
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 计算半径（近似）
                radius = int(np.sqrt(area / np.pi))
                
                # 检查中心点是否在图像边缘附近
                h_img, w_img = blurred.shape
                edge_margin = 30
                if center_x < edge_margin or center_x > w_img - edge_margin or \
                   center_y < edge_margin or center_y > h_img - edge_margin:
                    continue  # 边缘像素，跳过
                
                # 检查移动区域的形状（鼠标通常是圆形或小矩形）
                aspect_ratio = float(w) / h
                if 0.1 < aspect_ratio < 1.0:  # 允许一定的长宽比变化
                    # 更新前一帧
                    _prev_frame_gray = blurred.copy()
                    return center_x, center_y, radius
    
    # 更新前一帧
    _prev_frame_gray = blurred.copy()
    
    # 方法1: 使用普通阈值处理（针对测试图像中的白色圆形）
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤轮廓，寻找可能的鼠标指针
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤大小
        if min_size < area < max_size:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算宽高比（鼠标指针通常接近圆形，宽高比接近1）
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue  # 跳过非圆形的轮廓
            
            # 计算轮廓的圆形度（圆度）
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                # 鼠标指针通常比较圆
                if circularity < 0.7:
                    continue  # 跳过不圆的轮廓
            
            # 计算中心点
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 计算半径（近似）
            radius = int(np.sqrt(area / np.pi))
            
            # 检查中心点是否在图像边缘附近（边缘像素通常不是鼠标指针）
            h_img, w_img = blurred.shape
            edge_margin = 30
            if center_x < edge_margin or center_x > w_img - edge_margin or \
               center_y < edge_margin or center_y > h_img - edge_margin:
                continue  # 边缘像素，跳过
            
            return center_x, center_y, radius
    
    # 方法2: 使用自适应阈值处理（针对真实场景）
    binary2 = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours2:
        area = cv2.contourArea(contour)
        if min_size < area < max_size:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算宽高比（鼠标指针通常接近圆形，宽高比接近1）
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue  # 跳过非圆形的轮廓
            
            # 计算轮廓的圆形度（圆度）
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                # 鼠标指针通常比较圆
                if circularity < 0.7:
                    continue  # 跳过不圆的轮廓
            
            # 计算中心点
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 计算半径（近似）
            radius = int(np.sqrt(area / np.pi))
            
            # 检查中心点是否在图像边缘附近（边缘像素通常不是鼠标指针）
            h_img, w_img = blurred.shape
            edge_margin = 30
            if center_x < edge_margin or center_x > w_img - edge_margin or \
               center_y < edge_margin or center_y > h_img - edge_margin:
                continue  # 边缘像素，跳过
            
            return center_x, center_y, radius
    
    # 方法3: 直接寻找最亮的区域（鼠标指针通常比较亮）
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    if max_val > 245:  # 更高的亮度阈值，减少误检测
        # 以最亮点为中心，检查周围区域的亮度分布
        center_x, center_y = max_loc
        
        # 检查中心点周围的区域是否真的是一个小亮点（鼠标指针）
        # 获取中心点周围的区域
        h, w = blurred.shape
        roi_size = 10
        x_start = max(0, center_x - roi_size)
        x_end = min(w, center_x + roi_size)
        y_start = max(0, center_y - roi_size)
        y_end = min(h, center_y + roi_size)
        
        roi = blurred[y_start:y_end, x_start:x_end]
        
        # 计算ROI中的平均亮度
        mean_brightness = np.mean(roi)
        
        # 计算ROI中高亮像素的数量
        bright_pixels = np.sum(roi > 220)
        total_pixels = roi.size
        bright_ratio = bright_pixels / total_pixels
        
        # 鼠标指针通常是一个小的高亮区域，周围亮度较低
        # 高亮像素比例应该适中（不是太大也不是太小）
        if max_val - mean_brightness > 100 and 0.05 < bright_ratio < 0.3:
            # 检查高亮区域是否集中在中心
            # 计算中心区域的高亮像素比例
            center_roi_size = 5
            cx_start = max(0, center_x - center_roi_size)
            cx_end = min(w, center_x + center_roi_size)
            cy_start = max(0, center_y - center_roi_size)
            cy_end = min(h, center_y + center_roi_size)
            
            center_roi = blurred[cy_start:cy_end, cx_start:cx_end]
            center_bright_pixels = np.sum(center_roi > 220)
            center_total_pixels = center_roi.size
            center_bright_ratio = center_bright_pixels / center_total_pixels
            
            # 鼠标指针的高亮区域应该集中在中心
            if center_bright_ratio > bright_ratio * 1.5:
                radius = 10  # 默认半径
                return center_x, center_y, radius
    
    return None


def match_mouse_template(frame, template):
    """
    使用模板匹配在帧中查找鼠标指针
    
    参数:
        frame: 输入视频帧 (numpy数组)
        template: 鼠标模板图像 (numpy数组)
    
    返回:
        tuple: (x, y, radius) 鼠标指针位置和大小，如果未检测到返回 None
    """
    # 转换为灰度图
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # 获取模板的尺寸
    template_height, template_width = template_gray.shape
    
    # 执行模板匹配
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 设置匹配阈值
    threshold = 0.8
    if max_val >= threshold:
        # 计算中心点
        center_x = max_loc[0] + template_width // 2
        center_y = max_loc[1] + template_height // 2
        radius = int(np.sqrt(template_width * template_height) / 2)
        
        return center_x, center_y, radius
    
    return None


def track_mouse_pointer(video_path, output_path=None, frame_interval=1, show_preview=False, template=None):
    """
    跟踪视频中的鼠标指针轨迹
    
    参数:
        video_path: 输入视频路径
        output_path: 输出数据路径，如果为 None 则不保存
        frame_interval: 帧处理间隔，1=每帧处理
        show_preview: 是否显示实时预览
        template: 鼠标模板图像 (numpy数组)，如果提供则使用模板匹配
    
    返回:
        list: 鼠标轨迹数据列表，每个元素包含 (timestamp, x, y, radius)
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 帧处理间隔: {frame_interval}")
    if template is not None:
        print(f"  - 使用鼠标模板进行检测")
    
    # 初始化鼠标轨迹数据
    mouse_trajectory = []
    processed_frames = 0
    detected_frames = 0
    
    # 处理视频帧
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 跳过不需要处理的帧
        if frame_index % frame_interval != 0:
            frame_index += 1
            continue
        
        # 计算时间戳（秒）
        timestamp = frame_index / fps
        
        # 检测鼠标指针
        mouse_position = detect_mouse_pointer(frame, template=template)
        
        if mouse_position:
            x, y, radius = mouse_position
            mouse_trajectory.append({
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'radius': radius,
                'frame_index': frame_index
            })
            detected_frames += 1
            
            # 显示预览
            if show_preview:
                # 在帧上绘制鼠标指针
                cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
                cv2.putText(frame, f"Mouse: ({x}, {y})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_index}/{total_frames}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow('Mouse Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        processed_frames += 1
        frame_index += 1
    
    # 清理资源
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    # 保存数据
    if output_path:
        save_mouse_data(mouse_trajectory, output_path)
    
    # 统计信息
    print(f"\n处理完成!")
    print(f"  - 处理帧数: {processed_frames}")
    print(f"  - 检测到鼠标帧数: {detected_frames}")
    print(f"  - 鼠标检测率: {detected_frames/processed_frames*100:.2f}%")
    print(f"  - 总轨迹点: {len(mouse_trajectory)}")
    
    return mouse_trajectory


def process_screen_recording_with_template(video_path, template_paths, output_dir=None, frame_interval=1, show_preview=False):
    """
    使用鼠标模板处理屏幕录制视频，提取鼠标数据
    
    参数:
        video_path: 输入视频路径
        template_paths: 鼠标模板图像路径列表（可以是单张或多张）
        output_dir: 输出目录，如果为 None 则使用视频所在目录
        frame_interval: 帧处理间隔
        show_preview: 是否显示预览
    
    返回:
        dict: 处理结果
    """
    # 加载鼠标模板图像
    templates = []
    
    # 处理模板路径（支持单张或多张）
    if isinstance(template_paths, str):
        # 如果是单张模板
        template_paths = [template_paths]
    
    for template_path in template_paths:
        template = cv2.imread(template_path)
        if template is None:
            print(f"警告: 无法加载鼠标模板图像: {template_path}")
        else:
            templates.append(template)
            print(f"  - 加载模板: {template_path}")
    
    if not templates:
        raise ValueError(f"无法加载任何鼠标模板图像")
    
    video_path = Path(video_path)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = video_path.parent / 'mouse_data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{video_path.stem}_mouse_template_{timestamp}"
    
    # 处理视频，提取鼠标轨迹
    trajectory = track_mouse_pointer(
        video_path=str(video_path),
        output_path=str(output_dir / f"{base_name}.json"),
        frame_interval=frame_interval,
        show_preview=show_preview,
        template=templates
    )
    
    # 分析鼠标移动
    analysis = analyze_mouse_movement(trajectory)
    
    # 保存分析结果
    analysis_path = output_dir / f"{base_name}_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"  - 分析结果已保存: {analysis_path}")
    
    # 打印分析结果
    print("\n鼠标移动分析:")
    print(f"  - 总轨迹点: {analysis.get('total_points', 0)}")
    print(f"  - 总移动距离: {analysis.get('total_distance', 0):.2f} 像素")
    print(f"  - 平均速度: {analysis.get('average_speed', 0):.2f} 像素/秒")
    print(f"  - 最大速度: {analysis.get('max_speed', 0):.2f} 像素/秒")
    print(f"  - 持续时间: {analysis.get('time_duration', 0):.2f} 秒")
    
    return {
        'trajectory': trajectory,
        'analysis': analysis,
        'output_dir': str(output_dir)
    }


def save_mouse_data(trajectory, output_path):
    """
    保存鼠标轨迹数据
    
    参数:
        trajectory: 鼠标轨迹数据列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据文件扩展名选择保存格式
    ext = output_path.suffix.lower()
    
    if ext == '.json':
        # 保存为 JSON 格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)
        print(f"  - 数据已保存为 JSON: {output_path}")
        
    elif ext == '.csv':
        # 保存为 CSV 格式
        if trajectory:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = trajectory[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trajectory)
            print(f"  - 数据已保存为 CSV: {output_path}")
        
    else:
        # 默认保存为 JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)
        print(f"  - 数据已保存为 JSON: {json_path}")


def analyze_mouse_movement(trajectory):
    """
    分析鼠标移动数据
    
    参数:
        trajectory: 鼠标轨迹数据列表
    
    返回:
        dict: 分析结果
    """
    if not trajectory:
        return {}
    
    # 计算总移动距离
    total_distance = 0
    speeds = []
    
    for i in range(1, len(trajectory)):
        prev = trajectory[i-1]
        curr = trajectory[i]
        
        # 计算距离
        distance = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
        total_distance += distance
        
        # 计算时间差
        time_diff = curr['timestamp'] - prev['timestamp']
        if time_diff > 0:
            speed = distance / time_diff
            speeds.append(speed)
    
    # 计算统计信息
    analysis = {
        'total_points': len(trajectory),
        'total_distance': total_distance,
        'average_speed': np.mean(speeds) if speeds else 0,
        'max_speed': max(speeds) if speeds else 0,
        'min_speed': min(speeds) if speeds else 0,
        'time_duration': trajectory[-1]['timestamp'] - trajectory[0]['timestamp'] if len(trajectory) > 1 else 0
    }
    
    return analysis


def process_screen_recording(video_path, output_dir=None, frame_interval=1, show_preview=False):
    """
    处理屏幕录制视频，提取鼠标数据
    
    参数:
        video_path: 输入视频路径
        output_dir: 输出目录，如果为 None 则使用视频所在目录
        frame_interval: 帧处理间隔
        show_preview: 是否显示预览
    
    返回:
        dict: 处理结果
    """
    video_path = Path(video_path)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = video_path.parent / 'mouse_data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{video_path.stem}_mouse_{timestamp}"
    
    # 处理视频，提取鼠标轨迹
    trajectory = track_mouse_pointer(
        video_path=str(video_path),
        output_path=str(output_dir / f"{base_name}.json"),
        frame_interval=frame_interval,
        show_preview=show_preview
    )
    
    # 分析鼠标移动
    analysis = analyze_mouse_movement(trajectory)
    
    # 保存分析结果
    analysis_path = output_dir / f"{base_name}_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"  - 分析结果已保存: {analysis_path}")
    
    # 打印分析结果
    print("\n鼠标移动分析:")
    print(f"  - 总轨迹点: {analysis.get('total_points', 0)}")
    print(f"  - 总移动距离: {analysis.get('total_distance', 0):.2f} 像素")
    print(f"  - 平均速度: {analysis.get('average_speed', 0):.2f} 像素/秒")
    print(f"  - 最大速度: {analysis.get('max_speed', 0):.2f} 像素/秒")
    print(f"  - 持续时间: {analysis.get('time_duration', 0):.2f} 秒")
    
    return {
        'trajectory': trajectory,
        'analysis': analysis,
        'output_dir': str(output_dir)
    }


def visualize_mouse_trajectory(video_path, trajectory_path, output_video=None, show_preview=False):
    """
    可视化鼠标轨迹
    
    参数:
        video_path: 原始视频路径
        trajectory_path: 鼠标轨迹数据路径
        output_video: 输出视频路径
        show_preview: 是否显示预览
    """
    # 加载鼠标轨迹数据
    with open(trajectory_path, 'r', encoding='utf-8') as f:
        trajectory = json.load(f)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 准备输出视频
    output_writer = None
    if output_video:
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_index = 0
    trajectory_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 绘制鼠标轨迹
        for i in range(trajectory_index, len(trajectory)):
            point = trajectory[i]
            if point['frame_index'] <= frame_index:
                x, y = point['x'], point['y']
                radius = point['radius']
                
                # 绘制鼠标指针
                cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
                
                # 绘制轨迹线
                if i > 0:
                    prev_point = trajectory[i-1]
                    if prev_point['frame_index'] == frame_index - 1:
                        prev_x, prev_y = prev_point['x'], prev_point['y']
                        cv2.line(frame, (prev_x, prev_y), (x, y), (0, 255, 0), 2)
                
                trajectory_index = i
            else:
                break
        
        # 显示信息
        cv2.putText(frame, f"Frame: {frame_index}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Trajectory Points: {trajectory_index}/{len(trajectory)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存到输出视频
        if output_writer:
            output_writer.write(frame)
        
        # 显示预览
        if show_preview:
            cv2.imshow('Mouse Trajectory Visualization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_index += 1
    
    # 清理资源
    cap.release()
    if output_writer:
        output_writer.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"  - 轨迹可视化完成")
    if output_video:
        print(f"  - 输出视频已保存: {output_video}")


def detect_mouse_in_images(image_paths, template_path=None):
    """
    检测多张图片中的鼠标指针
    
    参数:
        image_paths: 图片路径列表
        template_path: 鼠标模板图像路径或路径列表（可选）
    
    返回:
        dict: 每张图片的鼠标检测结果
    """
    results = {}
    
    # 如果提供了模板，加载模板
    templates = []
    
    # 处理模板路径（支持单张或多张）
    if template_path is not None:
        if isinstance(template_path, str):
            # 单张模板
            template_paths = [template_path]
        else:
            # 多张模板
            template_paths = template_path
        
        for temp_path in template_paths:
            template = cv2.imread(temp_path)
            if template is None:
                print(f"警告: 无法加载模板图像: {temp_path}")
            else:
                templates.append(template)
                print(f"  - 加载模板: {temp_path}")
        
        if templates:
            print(f"共加载 {len(templates)} 张模板图像")
    
    for image_path in image_paths:
        print(f"\n处理图片: {image_path}")
        
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ❌ 无法加载图片: {image_path}")
            results[image_path] = None
            continue
        
        # 检测鼠标
        mouse_position = detect_mouse_pointer(image, template=templates if templates else None)
        
        if mouse_position:
            x, y, radius = mouse_position
            print(f"  ✅ 检测到鼠标指针:")
            print(f"    位置: ({x}, {y})")
            print(f"    半径: {radius}")
            
            # 在图片上绘制鼠标指针
            cv2.circle(image, (x, y), radius, (0, 0, 255), 2)
            cv2.putText(image, f"Mouse: ({x}, {y}) Radius: {radius}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存结果图片
            result_image_path = f"result_{Path(image_path).name}"
            cv2.imwrite(result_image_path, image)
            print(f"  📷 结果已保存: {result_image_path}")
            
            results[image_path] = {
                'position': (x, y),
                'radius': radius,
                'result_image': result_image_path
            }
        else:
            print(f"  ❌ 未检测到鼠标指针")
            results[image_path] = None
    
    return results


def compare_mouse_positions(results):
    """
    比较多张图片中的鼠标位置
    
    参数:
        results: 鼠标检测结果字典
    """
    print("\n=== 鼠标位置比较 ===")
    
    # 收集有效的检测结果
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print("  检测到的鼠标位置不足，无法比较")
        return
    
    # 计算所有位置之间的距离
    image_paths = list(valid_results.keys())
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            img1 = image_paths[i]
            img2 = image_paths[j]
            
            pos1 = valid_results[img1]['position']
            pos2 = valid_results[img2]['position']
            
            # 计算距离
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            print(f"  {Path(img1).name} → {Path(img2).name}:")
            print(f"    位置1: {pos1}")
            print(f"    位置2: {pos2}")
            print(f"    距离: {distance:.2f} 像素")


def detect_mouse_by_frame_diff(video_path, output_dir=None, frame_interval=1):
    """
    依据帧差别找到鼠标轮廓并输出图片
    
    参数:
        video_path: 输入视频路径
        output_dir: 输出目录，如果为 None 则使用视频所在目录
        frame_interval: 帧处理间隔
    
    返回:
        list: 鼠标轮廓图片路径列表
    """
    print("=== 依据帧差别检测鼠标轮廓 ===")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 帧处理间隔: {frame_interval}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(video_path).parent / 'mouse_contours'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - 输出目录: {output_dir}")
    
    # 初始化变量
    prev_frame_gray = None
    contour_images = []
    processed_frames = 0
    detected_frames = 0
    
    # 鼠标位置跟踪（用于运动连续性分析）
    prev_mouse_pos = None
    pos_history = []
    
    # 处理视频帧
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 跳过不需要处理的帧
        if frame_index % frame_interval != 0:
            frame_index += 1
            continue
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 帧差距分析
        if prev_frame_gray is not None:
            # 计算当前帧与前一帧的差异
            frame_diff = cv2.absdiff(blurred, prev_frame_gray)
            
            # 对差异图像进行阈值处理（提高阈值减少误检测）
            _, diff_binary = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)
            
            # 形态学操作，去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel)
            diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤轮廓，寻找可能的鼠标移动
            mouse_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 过滤大小（更严格的范围）
                if 0 < area < 300:
                    # 计算轮廓的边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 检查中心点是否在图像边缘附近
                    edge_margin = 50
                    if x > edge_margin and x + w < width - edge_margin and \
                       y > edge_margin and y + h < height - edge_margin:
                        # 检查移动区域的形状（更严格的宽高比）
                        aspect_ratio = float(w) / h
                        if 0.01 < aspect_ratio < 1.5:
                            # 检查轮廓的圆度
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                                if 0.3 < circularity < 0.9:
                                    mouse_contours.append(contour)
            
            # 为每一帧生成轮廓图片
            # 创建当前帧的副本用于绘制
            contour_frame = frame.copy()
            
            # 初始化检测状态
            detected = False
            detection_info = "未检测到鼠标"
            
            # 运动连续性分析：选择最可能的鼠标轮廓
            best_contour = None
            best_score = 0
            
            if mouse_contours:
                for contour in mouse_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    current_pos = (center_x, center_y)
                    
                    # 计算与之前位置的距离
                    score = 1.0
                    if prev_mouse_pos:
                        distance = math.sqrt((center_x - prev_mouse_pos[0]) ** 2 + 
                                           (center_y - prev_mouse_pos[1]) ** 2)
                        # 鼠标移动距离通常在合理范围内
                        if distance < 200:  # 最大移动距离
                            # 距离越近，得分越高
                            score += 1.0 / (1.0 + distance / 50)
                    
                    # 轮廓大小得分
                    if 100 < cv2.contourArea(contour) < 250:
                        score += 0.5
                    
                    # 形状得分
                    aspect_ratio = float(w) / h
                    if 0.8 < aspect_ratio < 1.2:
                        score += 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_contour = contour
                
                # 如果找到最佳轮廓
                if best_contour is not None and best_score > 1.2:  # 阈值，确保有一定可信度
                    # 计算最佳轮廓的位置
                    x, y, w, h = cv2.boundingRect(best_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 更新鼠标位置历史
                    prev_mouse_pos = (center_x, center_y)
                    pos_history.append(prev_mouse_pos)
                    if len(pos_history) > 10:  # 保留最近10个位置
                        pos_history.pop(0)
                    
                    # 在帧上绘制轮廓
                    cv2.drawContours(contour_frame, [best_contour], -1, (0, 0, 255), 2)
                    
                    # 更新检测信息
                    detected = True
                    detection_info = f"检测到鼠标轮廓 (置信度: {best_score:.2f})"
                    
                    # 添加文本信息
                    cv2.putText(contour_frame, f"Mouse Contour Detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(contour_frame, f"Position: ({center_x}, {center_y}) Frame: {frame_index}/{total_frames}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(contour_frame, f"Confidence: {best_score:.2f}", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 绘制历史轨迹
                    if len(pos_history) > 1:
                        for i in range(1, len(pos_history)):
                            cv2.line(contour_frame, pos_history[i-1], pos_history[i], (0, 255, 255), 2)
                    
                    detected_frames += 1
            
            # 为每一帧添加基本信息
            cv2.putText(contour_frame, detection_info, 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(contour_frame, f"Frame: {frame_index}/{total_frames}", 
                       (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存轮廓图片（每一帧都保存）
            output_path = output_dir / f"mouse_contour_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), contour_frame)
            contour_images.append(str(output_path))
            
            # 打印信息
            print(f"  📷 帧 {frame_index}: {detection_info}")
        
        # 更新前一帧
        prev_frame_gray = blurred.copy()
        processed_frames += 1
        frame_index += 1
    
    # 清理资源
    cap.release()
    
    # 统计信息
    print(f"\n处理完成!")
    print(f"  - 处理帧数: {processed_frames}")
    print(f"  - 检测到鼠标轮廓帧数: {detected_frames}")
    print(f"  - 生成轮廓图片: {len(contour_images)}")
    print(f"  - 输出目录: {output_dir}")
    
    return contour_images


if __name__ == "__main__":
    # 选项1: 处理视频文件（默认检测方法）
    process_video = False  # 是否处理视频
    video_path = r'E:\数据\20231229 计算机网络考试数据汇总\第1组\录屏\2021214387_周婉婷.mp4'  # 输入视频路径
    template_paths = [
        r"D:\Pycharm_Projects\demo1_trae\1.png",
        r"D:\Pycharm_Projects\demo1_trae\2.png",
        r"D:\Pycharm_Projects\demo1_trae\5.png",
    ]  # 鼠标模板图像路径列表（使用模板匹配）
    output_dir = None  # 输出目录
    frame_interval = 1  # 帧处理间隔
    show_preview = True  # 是否显示预览
    visualize = True  # 是否生成可视化视频
    
    # 选项2: 依据帧差别检测鼠标轮廓
    process_frame_diff = True  # 是否使用帧差距分析
    frame_diff_output_dir =r"D:\Pycharm_Projects\demo1_trae\output\diff"   # 帧差距分析输出目录
    frame_diff_interval = 1  # 帧差距分析处理间隔
    
    # 选项3: 处理多张图片
    process_images = False  # 是否处理图片
    image_paths = [
        r"D:\Pycharm_Projects\demo1_trae\3.png",
         r"D:\Pycharm_Projects\demo1_trae\4.png"
    ]
    image_template_path = None  # 图片检测使用的模板
    
    # 处理视频
    if process_video:
        print("=== 处理视频文件 ===")
        if template_paths:
            # 使用模板匹配
            result = process_screen_recording_with_template(
                video_path=video_path,
                template_paths=template_paths,
                output_dir=output_dir,
                frame_interval=frame_interval,
                show_preview=show_preview
            )
        else:
            # 使用默认检测方法
            result = process_screen_recording(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval=frame_interval,
                show_preview=show_preview
            )
        
        # 如果需要可视化
        if visualize:
            from pathlib import Path
            video_path_obj = Path(video_path)
            output_dir_obj = Path(result['output_dir'])
            
            # 查找轨迹文件
            if template_paths:
                trajectory_files = list(output_dir_obj.glob(f"{video_path_obj.stem}_mouse_template_*.json"))
            else:
                trajectory_files = list(output_dir_obj.glob(f"{video_path_obj.stem}_mouse_*.json"))
            
            if trajectory_files:
                trajectory_path = trajectory_files[0]
                output_video = output_dir_obj / f"{video_path_obj.stem}_visualization.mp4"
                
                visualize_mouse_trajectory(
                    video_path=video_path,
                    trajectory_path=str(trajectory_path),
                    output_video=str(output_video),
                    show_preview=show_preview
                )

    # 依据帧差别检测鼠标轮廓
    if process_frame_diff:
        print("\n=== 依据帧差别检测鼠标轮廓 ===")
        # 调用帧差距分析函数
        contour_images = detect_mouse_by_frame_diff(
            video_path=video_path,
            output_dir=frame_diff_output_dir,
            frame_interval=frame_diff_interval
        )
        
        print(f"\n=== 帧差距分析完成 ===")
        print(f"  - 生成轮廓图片: {len(contour_images)}")
        if contour_images:
            print(f"  - 示例输出: {contour_images[0]}")

    # 处理图片
    if process_images:
        print("\n=== 处理图片文件 ===")
        # 检测图片中的鼠标
        image_results = detect_mouse_in_images(image_paths, template_path=image_template_path)
        
        # 比较鼠标位置
        compare_mouse_positions(image_results)
        

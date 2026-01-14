# 这是一个示例 Python 脚本。
from preprocessing import video_data_process
import cv2
import  conf

def main():
    # 处理单张图片
    image_path = r"12月20日.png"  # 替换为你的图片路径
    result_person = video_data_process.find_main_person(
        image_path=image_path,
        model_weights=conf.model,
        center_weight=0.6,
        size_weight=0.4,
        return_type='all'
    )
    result_face = video_data_process.find_main_face(
        image_path=image_path,
        model_weights=conf.face_model,
        center_weight=0.6,
        size_weight=0.4,
        return_type='all'
    )
    result = video_data_process.find_main_person_and_inpaint(
        image_path=image_path,
        model_weights=conf.seg_model,
        center_weight=0.6,
        size_weight=0.4,
        conf_threshold=0.25,
        min_person_ratio=0.01,
        padding_ratio=0,
        inpaint_method='ns',
        inpaint_radius=3,
        return_type='all',
        use_person_class_only=True
    )
    # 对关键人其他区域做模糊处理
    result = video_data_process.find_main_person_and_blur(
        image_path=image_path,
        model_weights=conf.seg_model,
        center_weight=0.6,
        size_weight=0.4,
        conf_threshold=0.25,
        min_person_ratio=0.01,
        blur_strength=200,  # 更强的模糊效果
        blur_method='gaussian',
        edge_smooth=True,
        edge_blur_radius=7,
        return_type='all',
        use_person_class_only=True
    )

    if result is not None:
        # 保存所有结果
        cv2.imwrite("processed_focused.jpg", result['processed'])
        cv2.imwrite("annotated_simple.jpg", result['annotated'])
        cv2.imwrite("main_person_mask.jpg", result['main_mask'])
        cv2.imwrite("blurred_background.jpg", result['blurred_bg'])

        print("处理完成！结果已保存。")


    # if result_person and result_face:
    #     cropped_person, annotated_img_person, person_info = result_person
    #     cropped_face, annotated_img_face, face_info = result_face
    #     cv2.imwrite("main_person_cropped.jpg", cropped_person)
    #     # cv2.imwrite("annotated_persons.jpg", annotated_img)
    #     cv2.imwrite("main_face_cropped.jpg", cropped_face)
    #     # cv2.imwrite("annotated_faces.jpg", annotated_img)
    #     print("已保存裁剪图和标注图")


# 简单用法示例
def process_video_example():
    """
    处理视频的简单示例
    """
    input_video = r"C:\Users\dengm\Videos\12月20日_1min-1.mp4"
    output_video = "output_video.mp4"

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
    result = video_data_process.process_video_with_blur(
        video_path=input_video,
        output_path=output_video,
        frame_interval=1,  # 处理每一帧
        target_fps=30,  # 输出30fps
        show_preview=True,  # 显示预览
        progress_bar=True,  # 显示进度条
        **blur_params
    )

    return result


if __name__ == '__main__':
    # main()
    # process_video_example()

    # 检查 TensorFlow 模块的属性
    import tensorflow as tf

    print(dir(tf))
    # 查看是否有 version 属性
    print(hasattr(tf, '__version__'))
    print(hasattr(tf, 'version'))

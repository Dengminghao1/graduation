
import cv2

# 打开视频文件
video_path = r"C:\Users\dengm\Desktop\openpose\examples\media\video.avi"
cap = cv2.VideoCapture(video_path)

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频总帧数（属性）: {total_frames}")
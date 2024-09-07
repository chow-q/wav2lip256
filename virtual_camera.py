import cv2
import pyvirtualcam
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 全局变量用于存储最新的视频文件
new_video_file = None
watch_folder = '/tmp'

# 处理文件变化事件的类
class VideoFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        global new_video_file
        # 如果新创建的文件是 .mp4 格式，则更新 new_video_file
        if event.is_directory:
            return  # 忽略目录
        if event.src_path.endswith('.mp4'):
            new_video_file = event.src_path
            print(f"检测到新视频文件：{new_video_file}")

# 播放视频函数
def play_video(video_path, cam):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return False

    while cap.isOpened():
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕")
            break

        # 将帧转换为 RGB 格式并发送到虚拟摄像头
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(frame_rgb)

        # 如果有新的视频文件则切换播放
        global new_video_file
        if new_video_file:
            print(f"准备切换到新视频：{new_video_file}")
            cap.release()  # 释放当前视频资源
            return new_video_file  # 返回新视频文件路径以进行切换

        # 显示当前帧 (可选)
        cv2.imshow('Virtual Camera Output', frame)

        # 等待1ms后继续播放下一个帧，按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    return None  # 当前视频播放完毕，无需切换

# 监控 /tmp 文件夹的函数
def monitor_folder():
    event_handler = VideoFileHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_folder, recursive=False)
    observer.start()
    print(f"开始监控文件夹：{watch_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 主程序
if __name__ == "__main__":
    # 启动文件夹监控的线程
    import threading
    monitor_thread = threading.Thread(target=monitor_folder, daemon=True)
    monitor_thread.start()

    # 初始默认视频文件
    default_video = 'test.mp4'

    # 主循环，持续检查并播放视频
    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        print(f'虚拟摄像头已启动: {cam.device}')

        while True:
            # 如果有新的视频，先播放新视频，否则播放默认视频
            if new_video_file:
                current_video = new_video_file
            else:
                current_video = default_video

            # 播放当前视频
            new_video = play_video(current_video, cam)

            # 播放完新视频后回到默认视频
            if new_video:
                new_video_file = None  # 重置新视频文件标记
            else:
                print(f"继续播放默认视频：{default_video}")

            # 等待1秒后重新检查
            time.sleep(1)

            # 显示窗口关闭条件
            if cv2.getWindowProperty('Virtual Camera Output', cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtGui import QImage
import os
import skill
import imghdr  # 用于图片方向检测


class VideoProcessor(QThread):
    # 信号：处理后的帧
    frame_processed = pyqtSignal(QImage)
    # 信号：聚集事件检测
    aggregation_detected = pyqtSignal()
    # 信号：追逐事件检测
    chasing_detected = pyqtSignal()
    # 信号：处理错误
    processing_error = pyqtSignal(str)
    # 信号：处理完成
    processing_finished = pyqtSignal()

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True
        self.paused = False
        self.analyzer = skill.FishBehaviorAnalyzer()
        self.is_image = False  # 标记是否为图片
        self.cap = None
        self.fps = 30  # 默认FPS
        self.is_video_file = False
        self.seek_mutex = QMutex()  # 用于保护跳转操作
        self.seek_requested = False
        self.seek_position = 0
        self.seek_direction = 0  # 0: 无, 1: 前进, -1: 后退

    def run(self):
        """主处理循环"""
        try:
            # 检查是否为图片
            if isinstance(self.source, str):
                ext = os.path.splitext(self.source)[1].lower()
                self.is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp']
                self.is_video_file = not self.is_image

            # 根据源类型创建捕获对象
            if self.is_image:
                # 处理单张图片
                frame = cv2.imread(self.source)
                if frame is None:
                    self.processing_error.emit(f"无法打开图片: {self.source}")
                    return

                # 自动校正图片方向 (仅对图片)
                frame = self.auto_orient_image(frame, self.source)

                # 处理帧并检测行为
                processed_frame, has_aggregation, has_chasing = self.analyzer.process_frame(frame)

                # 发射事件信号
                if has_aggregation:
                    self.aggregation_detected.emit()
                if has_chasing:
                    self.chasing_detected.emit()

                # 转换为QImage并发射
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                self.frame_processed.emit(qt_image)

                # 图片只需处理一次，但保持线程运行直到停止
                while self.running:
                    self.msleep(100)  # 避免CPU占用过高
            else:
                # 处理视频或摄像头
                if isinstance(self.source, int):
                    self.cap = cv2.VideoCapture(self.source)
                    # 为摄像头设置较小的缓冲区，减少延迟
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    self.cap = cv2.VideoCapture(self.source)
                    self.is_video_file = True
                    # 为视频文件设置合适的缓冲区
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

                if not self.cap.isOpened():
                    self.processing_error.emit(f"无法打开视频源: {self.source}")
                    return

                # 获取视频FPS
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

                while self.running:
                    # 处理暂停状态
                    if self.paused:
                        self.msleep(100)  # 暂停时减少CPU占用
                        continue

                    # 检查是否有跳转请求
                    if self.seek_requested:
                        self.handle_seek_request()

                    ret, frame = self.cap.read()
                    if not ret:
                        # 视频播放结束
                        break

                    # 处理帧并检测行为
                    processed_frame, has_aggregation, has_chasing = self.analyzer.process_frame(frame)

                    # 发射事件信号
                    if has_aggregation:
                        self.aggregation_detected.emit()
                    if has_chasing:
                        self.chasing_detected.emit()

                    # 转换为QImage并发射
                    rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                    self.frame_processed.emit(qt_image)

                    # 控制处理速度
                    delay = max(1, int(1000 / self.fps) - 10)  # 减去处理时间
                    self.msleep(delay)

                if self.cap:
                    self.cap.release()
        except Exception as e:
            self.processing_error.emit(f"处理错误: {str(e)}")
        finally:
            # 处理完成时发出信号
            self.processing_finished.emit()

    def handle_seek_request(self):
        """处理跳转请求"""
        self.seek_mutex.lock()
        try:
            if self.seek_requested and self.cap is not None:
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

                if self.seek_direction == 1:  # 前进
                    new_frame = current_frame + int(self.seek_position * self.fps)
                    if new_frame < total_frames:
                        # 使用更安全的跳转方式
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                elif self.seek_direction == -1:  # 后退
                    new_frame = max(0, current_frame - int(self.seek_position * self.fps))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

                # 清空缓冲区，避免旧帧干扰
                self.flush_buffer()

            self.seek_requested = False
            self.seek_direction = 0
            self.seek_position = 0
        except Exception as e:
            print(f"跳转错误: {e}")
        finally:
            self.seek_mutex.unlock()

    def flush_buffer(self):
        """清空视频缓冲区"""
        if self.cap:
            # 设置缓冲区大小为1，然后恢复
            buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 读取一帧来清空缓冲区
            ret, frame = self.cap.read()
            if ret:
                # 处理这一帧
                processed_frame, has_aggregation, has_chasing = self.analyzer.process_frame(frame)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_processed.emit(qt_image)
            # 恢复缓冲区大小
            if self.is_video_file:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def toggle_pause(self):
        """切换暂停状态"""
        self.paused = not self.paused

    def seek_forward(self, seconds):
        """前进指定秒数"""
        if self.is_video_file and self.cap is not None:
            self.seek_mutex.lock()
            try:
                self.seek_requested = True
                self.seek_direction = 1
                self.seek_position = seconds
            finally:
                self.seek_mutex.unlock()

    def seek_backward(self, seconds):
        """后退指定秒数"""
        if self.is_video_file and self.cap is not None:
            self.seek_mutex.lock()
            try:
                self.seek_requested = True
                self.seek_direction = -1
                self.seek_position = seconds
            finally:
                self.seek_mutex.unlock()

    def auto_orient_image(self, image, image_path):
        """自动校正图片方向"""
        try:
            # 检测图片方向
            orientation = self.get_image_orientation(image_path)

            if orientation == 3:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif orientation == 6:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        except Exception:
            # 如果无法检测方向，保持原图
            pass

        return image

    def get_image_orientation(self, image_path):
        """获取图片EXIF方向信息"""
        try:
            from PIL import Image
            img = Image.open(image_path)
            if hasattr(img, '_getexif'):
                exif = img._getexif()
                if exif:
                    # 方向标签ID是274
                    orientation = exif.get(274)
                    return orientation
        except:
            pass
        return 1  # 默认方向

    def stop(self):
        """停止处理"""
        self.running = False
        self.paused = False
        if self.cap:
            self.cap.release()
        self.wait(2000)  # 等待线程结束，最多2秒
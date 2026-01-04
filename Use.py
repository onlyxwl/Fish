import sys
import os
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSizePolicy, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
from view import VideoProcessor
import skill


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("鱼类行为分析系统")
        self.setGeometry(100, 100, 1200, 800)

        # 固定窗口大小
        self.setFixedSize(1200, 800)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 左侧控制面板 (20%)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(15)

        # 摄像头选择
        self.camera_label = QLabel("选择摄像头:")
        self.camera_combo = QComboBox()

        # 添加摄像头选项并附带说明
        self.camera_combo.addItem("主摄像头 (0 - 通常为内置)", 0)
        self.camera_combo.addItem("备用摄像头 (1 - 通常为外接)", 1)
        self.camera_combo.addItem("外部摄像头 (2)", 2)
        self.camera_combo.setToolTip("索引0: 系统默认主摄像头\n索引1: 第一个外接摄像头\n索引2: 第二个外接摄像头")

        self.realtime_btn = QPushButton("实时分析", self)
        self.realtime_btn.setFixedHeight(50)
        self.realtime_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.realtime_btn.clicked.connect(self.start_realtime)

        self.local_btn = QPushButton("本地文件", self)
        self.local_btn.setFixedHeight(50)
        self.local_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.local_btn.clicked.connect(self.open_local_file)

        # === 新增：视频控制按钮 ===
        control_layout = QHBoxLayout()

        self.pause_btn = QPushButton("暂停", self)
        self.pause_btn.setFixedHeight(40)
        self.pause_btn.setStyleSheet("font-size: 14px;")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)  # 初始不可用

        self.prev_btn = QPushButton("后退1秒", self)
        self.prev_btn.setFixedHeight(40)
        self.prev_btn.setStyleSheet("font-size: 14px;")
        self.prev_btn.clicked.connect(self.seek_backward)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("前进1秒", self)
        self.next_btn.setFixedHeight(40)
        self.next_btn.setStyleSheet("font-size: 14px;")
        self.next_btn.clicked.connect(self.seek_forward)
        self.next_btn.setEnabled(False)

        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)

        # 保存按钮
        self.save_btn = QPushButton("保存当前帧", self)
        self.save_btn.setFixedHeight(40)
        self.save_btn.setStyleSheet("font-size: 14px; background-color: #4CAF50; color: white;")
        self.save_btn.clicked.connect(self.save_current_frame)
        self.save_btn.setEnabled(False)

        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #555;")

        self.aggregation_label = QLabel("聚集事件: 0")
        self.aggregation_label.setStyleSheet("font-size: 14px; color: #C00;")

        self.chasing_label = QLabel("追逐事件: 0")
        self.chasing_label.setStyleSheet("font-size: 14px; color: #00C;")

        # 添加摄像头说明
        self.camera_info = QLabel(
            "<b>摄像头索引说明:</b><br>"
            "• 0: 系统默认主摄像头<br>"
            "• 1: 第一个外接摄像头<br>"
            "• 2: 第二个外接摄像头"
        )
        self.camera_info.setStyleSheet("font-size: 12px; color: #666; background-color: #f0f0f0; padding: 5px;")
        self.camera_info.setWordWrap(True)

        # 添加到左侧布局
        left_layout.addWidget(self.camera_label)
        left_layout.addWidget(self.camera_combo)
        left_layout.addWidget(self.realtime_btn)
        left_layout.addWidget(self.local_btn)
        left_layout.addLayout(control_layout)  # 添加控制按钮布局
        left_layout.addWidget(self.save_btn)  # 添加保存按钮
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)
        left_layout.addWidget(self.aggregation_label)
        left_layout.addWidget(self.chasing_label)
        left_layout.addStretch(1)
        left_layout.addWidget(self.camera_info)

        # 右侧视频显示区 (80%)
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #333; border: 1px solid #555;")
        # 设置大小策略 - 保持固定大小
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # 创建空图像用于清除显示
        self.clear_display()

        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(self.video_label, 8)

        self.setCentralWidget(main_widget)

        # 视频处理线程
        self.processor = None
        self.video_source = None
        self.is_processing = False
        self.current_frame = None  # 存储当前帧用于保存

        # 行为检测计数器
        self.aggregation_count = 0
        self.chasing_count = 0

    def clear_display(self):
        """清除视频显示区域"""
        # 创建空白图像
        blank = QPixmap(self.video_label.size())
        blank.fill(QColor("#333"))

        # 添加提示文字
        painter = QPainter(blank)
        painter.setPen(QColor("#888"))
        painter.setFont(self.font())
        painter.drawText(blank.rect(), Qt.AlignCenter, "请选择分析模式...")
        painter.end()

        self.video_label.setPixmap(blank)

    def start_realtime(self):
        """启动实时摄像头分析"""
        # 停止当前处理并清除显示
        self.stop_processing(clear_display=True)

        # 获取选中的摄像头索引
        camera_index = self.camera_combo.currentData()
        self.video_source = camera_index

        # 测试摄像头是否可用
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            QMessageBox.critical(self, "摄像头错误",
                                 f"无法打开摄像头 {camera_index}！请检查摄像头连接。")
            return
        cap.release()

        self.start_processing()
        self.status_label.setText(f"实时模式: 摄像头{camera_index}")

    def open_local_file(self):
        """打开本地文件进行分析"""
        # 停止当前处理并清除显示
        self.stop_processing(clear_display=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或图片文件",
            "",
            "媒体文件 (*.mp4 *.avi *.mov *.jpg *.png *.jpeg *.bmp)"
        )

        if file_path:
            self.video_source = file_path
            self.start_processing()
            self.status_label.setText(f"分析文件: {os.path.basename(file_path)}")
        else:
            # 用户取消选择时清除显示
            self.clear_display()
            self.status_label.setText("已取消选择")

    def start_processing(self):
        """启动视频处理线程"""
        if self.video_source is None:
            return

        self.processor = VideoProcessor(self.video_source)
        self.processor.frame_processed.connect(self.update_frame)
        self.processor.aggregation_detected.connect(self.handle_aggregation)
        self.processor.chasing_detected.connect(self.handle_chasing)
        self.processor.processing_error.connect(self.handle_processing_error)
        self.processor.processing_finished.connect(self.handle_processing_finished)
        self.processor.start()
        self.is_processing = True

        # 重置计数器
        self.aggregation_count = 0
        self.chasing_count = 0
        self.aggregation_label.setText("聚集事件: 0")
        self.chasing_label.setText("追逐事件: 0")

        # 启用控制按钮
        self.update_control_buttons(True)

    def update_control_buttons(self, enabled):
        """更新控制按钮状态"""
        is_video_file = isinstance(self.video_source, str) if self.video_source else False

        self.pause_btn.setEnabled(enabled)
        # 只有视频文件才允许前进后退
        self.prev_btn.setEnabled(enabled and is_video_file)
        self.next_btn.setEnabled(enabled and is_video_file)
        self.save_btn.setEnabled(enabled)

    def stop_processing(self, clear_display=False):
        """停止视频处理"""
        if self.processor:
            # 断开信号连接
            try:
                self.processor.frame_processed.disconnect(self.update_frame)
                self.processor.aggregation_detected.disconnect(self.handle_aggregation)
                self.processor.chasing_detected.disconnect(self.handle_chasing)
                self.processor.processing_error.disconnect(self.handle_processing_error)
                self.processor.processing_finished.disconnect(self.handle_processing_finished)
            except TypeError:
                # 如果连接不存在，忽略错误
                pass

            # 停止线程
            self.processor.stop()
            self.processor = None

        self.is_processing = False
        self.status_label.setText("已停止")

        # 禁用控制按钮
        self.update_control_buttons(False)

        # 重置暂停按钮文本
        self.pause_btn.setText("暂停")

        if clear_display:
            self.clear_display()

    def toggle_pause(self):
        """切换暂停/继续状态"""
        if self.processor and self.is_processing:
            self.processor.toggle_pause()
            if self.processor.paused:
                self.pause_btn.setText("继续")
                self.status_label.setText("已暂停")
            else:
                self.pause_btn.setText("暂停")
                self.status_label.setText("分析中...")

    def seek_forward(self):
        """前进1秒"""
        if self.processor and self.is_processing:
            self.processor.seek_forward(1)  # 前进1秒

    def seek_backward(self):
        """后退1秒"""
        if self.processor and self.is_processing:
            self.processor.seek_backward(1)  # 后退1秒

    def save_current_frame(self):
        """保存当前帧"""
        if self.current_frame is not None:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存当前帧",
                f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                "图片文件 (*.jpg *.png *.bmp)"
            )

            if file_path:
                # 保存当前帧
                success = self.current_frame.save(file_path)
                if success:
                    QMessageBox.information(self, "保存成功", f"图片已保存到: {file_path}")
                else:
                    QMessageBox.warning(self, "保存失败", "无法保存图片")
        else:
            QMessageBox.warning(self, "无法保存", "没有可用的帧数据")

    @pyqtSlot(QImage)
    def update_frame(self, image):
        """更新视频帧显示"""
        # 保存当前帧用于后续保存
        self.current_frame = image.copy()

        # 确保视频标签仍然存在
        if not self.video_label:
            return

        pixmap = QPixmap.fromImage(image)
        # 保持宽高比缩放以适应标签大小
        pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    @pyqtSlot()
    def handle_aggregation(self):
        """处理聚集事件"""
        self.aggregation_count += 1
        self.aggregation_label.setText(f"聚集事件: {self.aggregation_count}")

    @pyqtSlot()
    def handle_chasing(self):
        """处理追逐事件"""
        self.chasing_count += 1
        self.chasing_label.setText(f"追逐事件: {self.chasing_count}")

    @pyqtSlot(str)
    def handle_processing_error(self, message):
        """处理处理错误"""
        self.stop_processing(clear_display=True)
        QMessageBox.critical(self, "处理错误", message)

    @pyqtSlot()
    def handle_processing_finished(self):
        """处理完成时清除显示"""
        self.stop_processing(clear_display=True)

    def closeEvent(self, event):
        """窗口关闭时停止所有处理"""
        self.stop_processing(clear_display=True)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
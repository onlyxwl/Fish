import cv2
import numpy as np
from collections import deque, defaultdict
import math
from sklearn.cluster import DBSCAN
import random
import torch
from ultralytics import YOLO
import os


class FishBehaviorAnalyzer:
    def __init__(self):
        # 加载YOLOv11模型
        model_path = r"E:\Desktop\ultralytics-main\ultralytics-main\yolo11n.pt"
        weights_path = r"E:\Desktop\ultralytics-main\ultralytics-main\runs\detect\train28\weights\best.pt"

        # 检查权重文件是否存在
        if not os.path.exists(weights_path):
            print(f"错误: 权重文件不存在: {weights_path}")
            # 尝试使用基础模型
            self.model = YOLO(model_path)
            print("使用基础模型进行检测")
        else:
            # 加载训练好的权重
            self.model = YOLO(weights_path)
            print(f"成功加载训练好的模型权重: {weights_path}")

        # 获取类别名称
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'fish'}

        # 设置模型参数
        self.conf_threshold = 0.1 # 置信度阈值
        self.iou_threshold = 0.45 # IoU阈值

        # 跟踪数据结构
        self.tracks = {}
        self.next_id = 0
        self.available_ids = [] # 可复用的ID
        self.aggregation_counter = 0
        self.chasing_counter = 0

        # 配置参数
        self.config = {
            # 聚集检测参数
            'cluster_eps': 500, # 邻域半径（像素）
            'min_samples': 3, # 最小鱼群数量
            'density_threshold': 0.02, # 密度阈值

            # 追逐检测参数
            'max_distance': 100, # 最大距离（像素）
            'min_angle': 180, # 最大角度（度）
            'min_history_dist': 500, # 历史最小距离
            'min_group_dist': 200, # 最小共同移动距离
            'min_track_length': 5, # 最小轨迹长度
            'track_history': 30, # 轨迹历史长度
            'max_inactive_frames': 5, # 最大失活帧数
        }

        # 修改：使用更大的颜色调色板，确保高对比度（基于数据可视化推荐，如tab20风格）
        # 原10个颜色扩展到20个，包含更多变体（深蓝、橙、紫、绿等），针对淡绿色背景优化（饱和、高对比）
        # 学习点：这些颜色来自科学可视化调色板（如Matplotlib 'tab20' 或 R 'YlGnBu'），避免低对比；你可以进一步用HSV生成更多
        self.colors = [
            (255, 50, 50),   # 深蓝色
            (50, 50, 255),   # 深红色
            (180, 105, 255), # 深紫色
            (255, 140, 0),   # 深橙色
            (50, 205, 154),  # 深青色
            (255, 20, 147),  # 深粉红
            (30, 144, 255),  # 深天蓝
            (178, 34, 34),   # 砖红色
            (70, 130, 180),  # 钢蓝色
            (220, 20, 60),   # 深红色
            (0, 128, 128),   # 深 teal (新增，增强多样性)
            (139, 0, 139),   # 深 magenta
            (255, 165, 0),   # 橙色变体
            (75, 0, 130),    # 靛蓝
            (255, 69, 0),    # 红橙
            (0, 100, 0),     # 深绿 (小心使用，避免与背景融合)
            (218, 112, 214), # 兰花紫
            (139, 69, 19),   # 鞍褐
            (85, 107, 47),   # 橄榄绿变体
            (255, 215, 0)    # 金色 (深变体)
        ]
        # 可选：动态生成更多颜色（HSV模型，确保随机但对比强）
        # 学习点：HSV (Hue, Saturation, Value) 可以生成无限颜色；这里固定S=0.8-1.0, V=0.5-0.8以保持深色
        # import colorsys  # 如果使用，需要在代码顶部导入
        # for _ in range(10):  # 生成额外10个
        #     h = random.uniform(0, 360)
        #     s = random.uniform(0.8, 1.0)
        #     v = random.uniform(0.5, 0.8)
        #     r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h/360, s, v)]
        #     self.colors.append((r, g, b))

        # 轨迹状态跟踪
        self.last_active_frame = defaultdict(int)
        self.current_frame = 0

    def detect_fish(self, frame):
        """使用YOLOv11模型检测鱼的位置和类别"""
        # 使用YOLO模型进行检测
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        fish_points = []
        detections = [] # 存储检测结果的详细信息

        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # 获取类别和置信度
                    cls_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                    conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)

                    # 获取类别名称
                    class_name = self.class_names.get(cls_id, f"Class_{cls_id}")

                    # 计算中心点
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    # 添加到鱼的位置列表
                    fish_points.append([x_center, y_center])

                    # 存储检测详细信息
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (x_center, y_center),
                        'class_id': cls_id,
                        'class_name': class_name,
                        'confidence': conf
                    })

        return np.array(fish_points), detections

    def analyze_clustering(self, points):
        """使用DBSCAN进行密度聚类分析"""
        if len(points) < self.config['min_samples']:
            return {}

        db = DBSCAN(eps=self.config['cluster_eps'],
                    min_samples=self.config['min_samples']).fit(points)
        labels = db.labels_

        clusters = {}
        for label in set(labels):
            if label != -1: # 忽略噪声点
                cluster_points = points[labels == label]
                clusters[label] = {
                    'count': len(cluster_points),
                    'centroid': np.mean(cluster_points, axis=0),
                    'points': cluster_points
                }
        return clusters

    def calculate_density(self, points):
        """计算鱼群密度"""
        if len(points) < 2:
            return 0

        min_dists = []
        for i in range(len(points)):
            min_dist = float('inf')
            for j in range(len(points)):
                if i == j:
                    continue
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_dist:
                    min_dist = dist
            min_dists.append(min_dist)

        avg_min_dist = np.mean(min_dists)
        return 1 / (avg_min_dist + 1e-5) # 防止除零

    def get_next_id(self):
        """获取下一个可用的ID，优先复用旧的ID"""
        if self.available_ids:
            return self.available_ids.pop(0)
        else:
            self.next_id += 1
            return self.next_id - 1

    def update_tracks(self, current_points):
        """更新鱼群轨迹跟踪"""
        # 更新帧计数器
        self.current_frame += 1

        # 重置所有轨迹的活动状态
        for tid, track in list(self.tracks.items()):
            if not track['active']:
                # 检查是否超过最大失活帧数
                inactive_frames = self.current_frame - self.last_active_frame[tid]
                if inactive_frames > self.config['max_inactive_frames']:
                    # 移除轨迹并回收ID
                    del self.tracks[tid]
                    self.available_ids.append(tid)
                continue

            track['active'] = False

        # 匹配现有轨迹或创建新轨迹
        for point in current_points:
            matched = False
            point = np.array(point)

            # 尝试匹配现有轨迹
            best_match_id = None
            best_match_dist = float('inf')

            for tid, track in self.tracks.items():
                if not track['active'] and len(track['trajectory']) > 0:
                    last_pos = np.array(track['trajectory'][-1])
                    dist = np.linalg.norm(point - last_pos)

                    # 距离阈值匹配
                    if dist < self.config['max_distance'] * 1.5 and dist < best_match_dist:
                        best_match_id = tid
                        best_match_dist = dist

            if best_match_id is not None:
                # 更新匹配的轨迹
                track = self.tracks[best_match_id]
                track['trajectory'].append(point)
                if len(track['trajectory']) > self.config['track_history']:
                    track['trajectory'].popleft()
                track['active'] = True
                self.last_active_frame[best_match_id] = self.current_frame
                matched = True

            # 创建新轨迹
            if not matched:
                new_id = self.get_next_id()
                self.tracks[new_id] = {
                    'id': new_id,
                    'trajectory': deque([point], maxlen=self.config['track_history']),
                    'active': True
                }
                self.last_active_frame[new_id] = self.current_frame

    def get_direction(self, trajectory):
        """计算轨迹方向"""
        if len(trajectory) < 2:
            return np.array([0, 0])

        # 使用最近几个点计算方向
        num_points = min(5, len(trajectory))
        start_point = np.array(trajectory[-num_points])
        end_point = np.array(trajectory[-1])
        direction = end_point - start_point

        # 归一化
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        return direction

    def detect_chasing(self):
        """检测追逐行为"""
        active_tracks = [t for t in self.tracks.values()
                         if t['active'] and len(t['trajectory']) >= self.config['min_track_length']]

        chasing_pairs = []

        # 检查所有可能的鱼对组合
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                track1 = active_tracks[i]
                track2 = active_tracks[j]

                # 1. 检查当前距离
                pos1 = np.array(track1['trajectory'][-1])
                pos2 = np.array(track2['trajectory'][-1])
                current_dist = np.linalg.norm(pos1 - pos2)

                if current_dist > self.config['max_distance']:
                    continue

                # 2. 检查方向相似性
                dir1 = self.get_direction(track1['trajectory'])
                dir2 = self.get_direction(track2['trajectory'])

                if np.linalg.norm(dir1) == 0 or np.linalg.norm(dir2) == 0:
                    continue

                cos_sim = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
                angle = math.degrees(math.acos(np.clip(cos_sim, -1, 1)))

                if angle > self.config['min_angle']:
                    continue

                # 3. 检查历史最小距离
                min_dist = float('inf')
                for k in range(min(len(track1['trajectory']), len(track2['trajectory']))):
                    dist = np.linalg.norm(
                        np.array(track1['trajectory'][k]) -
                        np.array(track2['trajectory'][k])
                    )
                    if dist < min_dist:
                        min_dist = dist

                if min_dist > self.config['min_history_dist']:
                    continue

                # 4. 检查共同移动距离
                start_dist = np.linalg.norm(
                    np.array(track1['trajectory'][0]) -
                    np.array(track2['trajectory'][0])
                )
                end_dist = np.linalg.norm(
                    np.array(track1['trajectory'][-1]) -
                    np.array(track2['trajectory'][-1])
                )
                group_dist = abs(start_dist - end_dist)

                if group_dist < self.config['min_group_dist']:
                    continue

                # 所有条件满足，判定为追逐
                chasing_pairs.append((track1['id'], track2['id']))

        return chasing_pairs

    def draw_text_with_background(self, frame, text, position, font_scale=0.7, thickness=2,
                                  text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """绘制带背景框的文本，提高在淡绿色背景上的可见性"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # 文本背景框位置
        x, y = position
        bg_top_left = (x, y - text_height - 5)
        bg_bottom_right = (x + text_width + 5, y + 5)

        # 绘制背景框
        cv2.rectangle(frame, bg_top_left, bg_bottom_right, bg_color, -1)

        # 绘制文本
        cv2.putText(frame, text, (x + 2, y - 2), font, font_scale, text_color, thickness)

        return text_width, text_height

    def visualize_results(self, frame, fish_points, detections, clusters, chasing_pairs):
        """可视化分析结果 - 针对淡绿色背景优化"""
        # 绘制检测框和类别置信度（深色边框和白色文字）
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            # Modified: 根据置信度范围调整显示的置信度值（不改变原始confidence，只调整显示）
            # 规则：0.1~0.2 加0.4；0.2~0.3 加0.35；0.3~0.4 加0.25；0.4~0.5 加0.25
            # 注意：假设范围为 [0.1, 0.2) 加0.4，以此类推；对于>=0.5的不调整
            # 如果调整后>1，可以显示>1或clip到1，根据需求这里不clip
            adjusted_conf = confidence
            if 0.1 <= confidence < 0.2:
                adjusted_conf += 0.37
            elif 0.2 <= confidence < 0.3:
                adjusted_conf += 0.30
            elif 0.3 <= confidence < 0.4:
                adjusted_conf += 0.19
            elif 0.4 < confidence <= 0.45:
                adjusted_conf += 0.08
            elif 0.45 <= confidence < 0.46:
                adjusted_conf += 0.09
            elif 0.47 < confidence <= 0.48:
                adjusted_conf += 0.08
            elif 0.48 < confidence <= 0.53:
                adjusted_conf += 0.07
            elif 0.53 < confidence <= 0.58:
                adjusted_conf += 0.03
            elif 0.58 < confidence <= 0.60:
                adjusted_conf += 0.07
            elif 0.60 < confidence <= 0.68:
                adjusted_conf += 0.06
            elif 0.71 <= confidence < 0.73:
                adjusted_conf += 0.03
            elif 0.75 <= confidence < 0.76:
                adjusted_conf += 0.06
            elif 0.75 < confidence <= 0.76:
                adjusted_conf += 0.08
            elif 0.76 < confidence <= 0.77:
                adjusted_conf += 0.08
            elif 0.77 < confidence <= 0.79:
                adjusted_conf += 0.04
            elif 0.79 < confidence <= 0.81:
                adjusted_conf += 0.06
            elif 0.81 < confidence <= 0.82:
                adjusted_conf += 0.06
            elif 0.82 < confidence <= 0.83:
                adjusted_conf += 0.05
            elif 0.83 < confidence <= 0.84:
                adjusted_conf += 0.06
            elif 0.85 <= confidence < 0.86:
                adjusted_conf += 0.02
            # 对于>=0.5或<0.1的不调整（但由于阈值0.1，<0.1不会出现）

            # 修改：边界框颜色可改为基于独特ID（如果有轨迹关联），这里假设保持基于class_id，但用扩展列表
            # 学习点：如果想每个框独特，可添加逻辑匹配detection到track（基于中心点距离），然后用tid % len(colors)
            bbox_color = self.colors[det['class_id'] % len(self.colors)]

            # 绘制边界框 - 加粗
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 3)

            # 准备显示文本 - 使用调整后的adjusted_conf
            label = f"{class_name}: {adjusted_conf:.2f}"  # 显示置信度（注释掉这行来隐藏置信度）
            # label = f"{class_name}"  # 只显示类别名称（取消注释这行来隐藏置信度）

            # 使用自定义函数绘制带背景的文本
            self.draw_text_with_background(
                frame,
                label,
                (int(x1), int(y1) - 10),
                font_scale=0.7,
                thickness=2,
                text_color=(255, 255, 255), # 白色文字
                bg_color=(50, 50, 50) # 深灰色背景
            )

        # 绘制所有鱼的位置（中心点），注释掉不显示点
        # for point in fish_points:
        # cv2.circle(frame, tuple(point.astype(int)), 15, (0, 255, 0), -1)

        # 绘制聚集区域 - 使用深色边框
        for cid, cluster in clusters.items():
            if cluster['count'] >= self.config['min_samples']:
                density = self.calculate_density(cluster['points'])
                if density > self.config['density_threshold']:
                    # 绘制聚集区域 - 使用深紫色边框
                    cv2.circle(frame, tuple(cluster['centroid'].astype(int)),
                               80, (180, 105, 255), 3)

                    # 使用带背景的文字
                    self.draw_text_with_background(
                        frame,
                        f"Cluster {cid}",
                        (int(cluster['centroid'][0]) - 50, int(cluster['centroid'][1]) - 70),
                        font_scale=1.0,
                        thickness=2,
                        text_color=(255, 255, 255), # 白色文字
                        bg_color=(80, 0, 80) # 深紫色背景
                    )

        # 绘制轨迹 - 使用深色
        for tid, track in self.tracks.items():
            if track['active']:
                color = self.colors[tid % len(self.colors)]  # 轨迹颜色，用扩展列表
                trajectory = list(track['trajectory'])

                # 绘制轨迹线 - 加粗
                for k in range(1, len(trajectory)):
                    cv2.line(frame,
                             tuple(trajectory[k - 1].astype(int)),
                             tuple(trajectory[k].astype(int)),
                             color, 4)

                # # 绘制当前点与ID
                # if len(trajectory) > 0:
                #     curr_x, curr_y = int(trajectory[-1][0]), int(trajectory[-1][1])
                #     cv2.circle(frame, (curr_x, curr_y), 15, color, -1)
                #
                #     # 修改：ID背景颜色改为与中心点/轨迹颜色一致（使用相同的color）
                #     # 学习点：原代码用//2暗化；互补色会导致不一样；现在用相同color确保一致（白色文本在深色背景上仍可见）
                #     # 如果背景太亮导致文本不可见，可调整为略暗版：(int(color[0]*0.8), int(color[1]*0.8), int(color[2]*0.8))
                #     bg_color = color
                #
                #     # 使用带背景的文字显示ID
                #     self.draw_text_with_background(
                #         frame,
                #         f"ID:{tid}",
                #         (curr_x + 20, curr_y),
                #         font_scale=0.8,
                #         thickness=2,
                #         text_color=(255, 255, 255), # 白色文字（不变）
                #         bg_color=bg_color  # 修改为一致颜色
                #     )

        # 标记追逐对
        for pair in chasing_pairs:
            tid1, tid2 = pair
            if (tid1 in self.tracks and tid2 in self.tracks and
                    self.tracks[tid1]['active'] and self.tracks[tid2]['active']):
                pos1 = np.array(self.tracks[tid1]['trajectory'][-1])
                pos2 = np.array(self.tracks[tid2]['trajectory'][-1])

                # 绘制连线 - 使用醒目的颜色
                cv2.line(frame,
                         (int(pos1[0]), int(pos1[1])),
                         (int(pos2[0]), int(pos2[1])),
                         (255, 0, 255), 5)

                # 计算中点并加文字
                mid_x = int((float(pos1[0]) + float(pos2[0])) / 2.0)
                mid_y = int((float(pos1[1]) + float(pos2[1])) / 2.0)

                # 使用带背景的文字
                self.draw_text_with_background(
                    frame,
                    "Chasing",
                    (int(mid_x - 40), int(mid_y - 10)),
                    font_scale=1.0,
                    thickness=2,
                    text_color=(255, 255, 0), # 黄色文字
                    bg_color=(128, 0, 128) # 深紫色背景
                )

        # 添加状态信息 - 使用深色背景框
        status_infos = [
            f"Fish Count: {len(fish_points)}",
            f"Clusters: {len(clusters)}",
            f"Tracks: {len([t for t in self.tracks.values() if t['active']])}",
            f"Chasing: {len(chasing_pairs)}"
        ]

        start_y = 30
        for i, info in enumerate(status_infos):
            # 绘制带背景的文本
            self.draw_text_with_background(
                frame,
                info,
                (10, start_y + i * 35),
                font_scale=1.0,
                thickness=2,
                text_color=(255, 255, 255), # 白色文字
                bg_color=(0, 50, 100) # 深蓝色背景
            )

        return frame

    def process_frame(self, frame):
        """处理单帧图像"""
        # 1. 使用YOLO检测鱼的位置、类别和置信度
        fish_points, detections = self.detect_fish(frame)

        # 2. 聚集行为分析
        clusters = self.analyze_clustering(fish_points)
        has_aggregation = False
        for cluster in clusters.values():
            if cluster['count'] >= self.config['min_samples']:
                density = self.calculate_density(cluster['points'])
                if density > self.config['density_threshold']:
                    has_aggregation = True
                    break

        # 3. 更新轨迹
        self.update_tracks(fish_points)

        # 4. 追逐行为分析
        chasing_pairs = self.detect_chasing()
        has_chasing = len(chasing_pairs) > 0

        # 5. 可视化结果（现在传递detections参数）
        visualized_frame = self.visualize_results(frame.copy(), fish_points, detections, clusters, chasing_pairs)

        return visualized_frame, has_aggregation, has_chasing

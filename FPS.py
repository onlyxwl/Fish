import time
from ultralytics import YOLO

# 加载训练好的最佳模型
model = YOLO("E:/Desktop/ultralytics-main/ultralytics-main/runs/detect/train27/weights/best.pt")

# 进行推理并测算时间
source = "E:/Desktop/ultralytics-main/ultralytics-main/datasets/fish/train/images"  # 或使用验证集的一部分
start_time = time.time()
results = model(source, stream=True)  # 使用stream=True以提高测速准确性

# 计算处理帧数（这里以处理100帧为例）
frame_count = 0
for r in results:
    frame_count += 1
    if frame_count >= 100:
        break

end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"推理速度 FPS: {fps:.2f}")
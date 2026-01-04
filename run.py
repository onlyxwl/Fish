import time
from ultralytics import YOLO


# # 自定义回调函数
# def on_train_end(trainer):
#     # 在训练结束后自动调用验证方法
#     # 使用训练时记住的数据集配置，也可以在这里指定其他参数，如 data、batch 等
#     metrics = trainer.validator.metrics  # 验证指标已在训练过程中计算完毕
#
#     # 输出你需要的指标
#     print("\n=== 训练完成，模型关键指标 ===")
#     print(f"mAP50 (%): {metrics.box.map50 * 100:.2f}%")  # mAP@0.5
#     print(f"mAP50-95 (%): {metrics.box.map * 100:.2f}%")  # mAP@0.5:0.95
#     print(f"F1分数 (%): {metrics.box.f1 * 100:.2f}%")  # F1分数
#     print(f"召回率 R (%): {metrics.box.r * 100:.2f}%")  # 召回率
#     # 注意：FPS通常不在val()中直接计算，需要单独测试


# 加载预训练模型
model = YOLO('yolov10n.pt')

# # 将回调函数添加到模型中
# model.add_callback("on_train_end", on_train_end)

# 增强数据增强配置
augment_config = {
    # 基础增强
    'hsv_h': 0.015,  # 色调增强强度（0-0.1）
    'hsv_s': 0.7,  # 饱和度增强强度（0-1）
    'hsv_v': 0.4,  # 亮度增强强度（0-1）

    # 几何变换
    'degrees': 10.0,  # 旋转角度（度）
    'translate': 0.2,  # 平移比例
    'scale': 0.5,  # 缩放比例
    'shear': 0.1,  # 错切变形

    # 高级增强
    'flipud': 0.5,  # 上下翻转概率
    'fliplr': 0.5,  # 左右翻转概率
    'mosaic': 1.0,  # Mosaic增强概率
    'mixup': 0.2  # Mixup增强概率
}

# 使用原始图像尺寸进行训练
results = model.train(
    data="E:\\Desktop\\ultralytics-main\\ultralytics-main\\datasets\\fish\\fish.yaml",
    epochs=300,
    imgsz=640,  # 固定输入尺寸（若需要原始尺寸应设为-1）
    device=[0],  # 使用GPU
    workers=0,  # 关闭多线程加载
    batch=4,  # 批次大小
    cache=True,  # 开启图像缓存

    # 学习率配置（余弦退火）
    lr0=1e-3,  # 初始学习率（注意：1e-3是更合理的值）
    lrf=0.01,  # 最终学习率（cosine退火终点）
    cos_lr=True,  # 启用余弦退火

    # 数据增强配置
    **augment_config
)

time.sleep(10)  # 训练完成后等待

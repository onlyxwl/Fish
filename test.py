from ultralytics import YOLO
import torch

# 加载模型（默认会使用 GPU，如果可用）
model = YOLO("yolo11.yaml")

# 获取当前设备（自动判断是否使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# 将模型显式移动到设备上（虽然YOLO默认会处理，但显式操作更安全）
model.to(device)

# 创建 dummy 输入，并将其放到相同的设备上
dummy_input = torch.randn(1, 3, 640, 640).to(device)

# 方法1：使用 model 推理接口（内部已处理设备问题）
results = model(dummy_input)

# 打印检测结果
print("\n推理结果:")
for i, r in enumerate(results):
    print(f"结果 {i}:")
    if r.boxes is not None:
        print(f"  检测框: {r.boxes.xyxy.shape}")
    if r.masks is not None:
        print(f"  分割掩码: {r.masks.data.shape}")
    if r.probs is not None:
        print(f"  分类概率: {r.probs.shape}")

# 方法2：直接调用模型结构（必须保证输入在相同设备）
with torch.no_grad():
    output = model.model(dummy_input)

# 打印输出张量形状
print("\n模型原始输出:")
if isinstance(output, (list, tuple)):
    for i, o in enumerate(output):
        if isinstance(o, (list, tuple)):
            print(f"输出 {i} 是一个列表，包含 {len(o)} 个元素:")
            for j, oo in enumerate(o):
                if isinstance(oo, torch.Tensor):
                    print(f"  元素 {j} 形状: {oo.shape}")
                else:
                    print(f"  元素 {j} 类型不为 Tensor: {type(oo)}")
        elif isinstance(o, torch.Tensor):
            print(f"输出 {i} 形状: {o.shape}")
        else:
            print(f"输出 {i} 类型不为 Tensor 或列表: {type(o)}")
else:
    if isinstance(output, torch.Tensor):
        print(f"输出形状: {output.shape}")
    else:
        print(f"输出类型不为 Tensor: {type(output)}")
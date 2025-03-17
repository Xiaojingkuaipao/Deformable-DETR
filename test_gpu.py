import torch

# 检查是否有可用的GPU
if torch.cuda.is_available():
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"当前可用的GPU数量: {gpu_count}")

    # 输出每个GPU的型号
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("没有可用的GPU，PyTorch将在CPU上运行。")
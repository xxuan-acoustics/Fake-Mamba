# import torch

# # 指定路径
# model_path = "/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/models/Exp19/best.pth"

# # 加载模型参数
# model_state_dict = torch.load(model_path, map_location='cuda:2')

# # 打印模型最后一层的维度
# # 假设最后一层是全连接层或类似的层，我们可以通过键值找到该层
# # 通常最后一层的权重键是 'fc.weight' 或其他类似名称，根据实际情况做调整
# last_layer_key = list(model_state_dict.keys())[-1]  # 获取最后一个层的键
# last_layer_weight = model_state_dict[last_layer_key]

# # 打印最后一层的维度
# print(f"The last layer's dimension is: {last_layer_weight.shape}")


import torch

# 指定路径
# model_path = "/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/models/Exp19/best/best_0.pth"
model_path = "/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/models/Exp41/epoch_0.pth"

# 加载模型参数
model_state_dict = torch.load(model_path, map_location='cuda:2')

# 打印每一层的名称和维度
print("Model Layers and their dimensions:")
for key, value in model_state_dict.items():
    print(f"{key}: {value.shape}")

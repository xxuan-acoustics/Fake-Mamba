import torch
import torch.nn as nn
import fairseq

class SSLModel(nn.Module):  # W2V
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = './pre-model/xlsr2_300m.pt'  # 修改为预训练模型的路径
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].to(device)  # 确保模型加载到指定设备
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # 确保模型在正确的设备上
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device)
            self.model.eval()  # 切换到评估模式

        # 将输入传递给特征提取器以获得适当的形状
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        with torch.no_grad():
            # 使用模型的特征提取器
            features = self.model.feature_extractor(input_tmp)
            features = features.transpose(1, 2)  # 转置以符合编码器输入形状 [batch, seq_len, feature_dim]
            
            # 如果有 post_extract_proj，则使用它将特征投影到 1024 维
            if self.model.post_extract_proj is not None:
                features = self.model.post_extract_proj(features)

            # 输出特征
            outputs = []
            for i, layer in enumerate(self.model.encoder.layers):
                features = layer(features)[0]  # 处理每个 transformer 层
                layer_output = layer.final_layer_norm(features)  # 提取 final_layer_norm 后的特征
                print(f"Layer {i} final_layer_norm output shape: {layer_output.shape}")  # 打印输出形状
                outputs.append(layer_output)
            return torch.stack(outputs, dim=1)  # 将每层的特征堆叠到一起

    def save_model_layers_info(self, file_path='SSL_layers.txt'):
        # 将模型的各层信息保存到文件
        with open(file_path, 'w') as f:
            f.write("Model Layers:\n")
            for name, layer in self.model.named_modules():
                layer_info = f"Layer: {name} | Type: {type(layer)}\n"
                f.write(layer_info)
                print(layer_info)  # 同时打印到控制台

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_model = SSLModel(device)

    # 保存模型层的名称到文件
    ssl_model.save_model_layers_info()

    # 示例：提取特征
    input_data = torch.randn(32, 66800).to(device)  # 假设是一个批次为32的音频输入
    features = ssl_model.extract_feat(input_data)
    print("Extracted feature shape:", features.shape)

if __name__ == "__main__":
    main()

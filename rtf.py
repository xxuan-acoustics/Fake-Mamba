# import time
# import torchaudio
# import torch
# import matplotlib.pyplot as plt
# import argparse
# import sys
# import os
# from torch import nn
# from torch.utils.data import DataLoader
# from data_utils import Dataset_train, Dataset_fixed_eval #Fixed
# from model import *
# import matplotlib.ticker as ticker

# def load_audio_segment(file_path, duration):
#     """
#     从完整音频中截取duration秒的音频片段
#     假设音频为单声道，16kHz采样率（如实际数据不符请根据需要重采样）。
#     """
#     waveform, sr = torchaudio.load(file_path)
#     # print("waveform",waveform)
#     # print(waveform.size(1))
#     target_length = int(sr * duration)
    
#     if waveform.size(1) > target_length:
#         # 截取前duration秒
#         waveform = waveform[:, :target_length]
#     else:
#         # 如果音频短于所需长度，可以对其进行零填充
#         pad_length = target_length - waveform.size(1)
#         waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
#     return waveform, sr

# def model_inference(waveform, model, device):
#     """
#     模型推理函数：将端到端模型应用于输入波形数据。
#     waveform形状假设为[channels, time]，根据你的模型需要可能进行变形。
#     """
#     # 假设模型需要 [batch, time] 且为单声道
#     waveform_input = waveform.squeeze(0).unsqueeze(0).to(device)  # [1, time] 并迁移到device
    
#     # 推理
#     with torch.no_grad():
#         output = model(waveform_input)  # 假设 model(...) 直接输出结果
#     return output

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Conformer-W2V')
#     # Dataset
#     parser.add_argument('--database_path', type=str, default='./datasets/cleaned/', help='Path to datasets')
#     parser.add_argument('--protocols_path', type=str, default='./database/', help='Path to protocols')
#     parser.add_argument('--exp_num', type=int, default=84, help='Experiment number for saving results')
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--num_epochs', type=int, default=100)
#     parser.add_argument('--lr', type=float, default=0.000001)
#     parser.add_argument('--weight_decay', type=float, default=0.0001)
#     parser.add_argument('--loss', type=str, default='WCE')

#     #model parameters
#     parser.add_argument('--emb-size', type=int, default=144, metavar='N',
#                         help='embedding size')
#     parser.add_argument('--heads', type=int, default=4, metavar='N',
#                         help='heads of the conformer encoder')
#     parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
#                         help='kernel size conv module')
#     parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
#                         help='number of encoders of the conformer')
#     parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']))

#     # model save path and evaluation parameters
#     parser.add_argument('--seed', type=int, default=1234, help='random seed')
#     parser.add_argument('--comment', type=str, default=None)
#     parser.add_argument('--comment_eval', type=str, default=None)
#     parser.add_argument('--train', default='true', type=lambda x: (str(x).lower() in ['true', 'yes', '1']))
#     parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
#     parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']))
#     parser.add_argument('--n_average_model', default=5, type=int)

#     args = parser.parse_args()
#     print(args)

#     # 主程序：对不同秒长的语音计算RTF
#     audio_file = "/home/xxuan/speech-deepfake/release_in_the_wild/1.wav"  # 请换成实际音频文件路径
#     utterance_durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]  # 修改后的语音片段时长（秒）

#     #GPU device
#     device = 'cuda:1' if torch.cuda.is_available() else 'cpu'                  
#     print('Device: {}'.format(device))

#     # 初始化模型
#     # model = Model(args,device)#baseline,37DF(EXP9,16LA_FIXED)(EXP20DF_FIXED)

#     # model = XLSR_BiMamba_CLS(args,device)#80LA,81DF

#     # model = XLSR_BiMamba_CLS(args,device)#82LA,83DF-MyBiMamba-_get_clones(BiMambaEncoder

#     # model = Model2(args,device)#baseline-val_test(EXP9,16LA_VAR)(EXP20DF_VAR)

#     # model = TCM_Model(args,device)#TCM

#     # model = ConBiMamba_Model(args,device)#exp15

#     # model = Model(args,device)#baseline-exp16,conformer00

#     # model = ConBiMamba_Model00(args,device)#exp17(conformer4,head4),18(conformer17,head4),17和18没有用MHSA，所以head没有传入
#     # 19(FFN+BIMAMBA+MHSA+CONV+FFN,conformer17,head4),34,36,38DF
    

#     # model = ConBiMamba_Model00_CLS(args,device)#exp23,24

#     # model = ConBiMamba_Model00_CLS_ASP(args,device)#exp25,28

#     # model = SSL_Model00_CLS_ASP(args,device)#exp26,27,35

#     # model = ASP_ConBiMamba_Model00(args,device)#EXP

#     # model = ConBiMamba_Model00_var(args,device)#exp18_test_val

#     # model = ConBiMamba_AM_Model(args,device)

#     # model = ConBiMamba_Model00_mutil_ASP(args,device)#EXP29,conformer_block=4

#     # model = SSL_24output_ConBiMamba_Model00(args,device)#EXP32

#     # model = (args,device)#exp

#     # model = XLSX_SLS_Model(args,device)#40DF,42LA,43LA

#     # model = XLSX_MLLA_Model(args,device)#LA

#     # model = XLSX_CBAM_SLS_Model(args,device)#56LA

#     # model = XLSX_Mamba1_Model(args,device)#41LA,44LA,45LA(conbimamba,删掉conv+ffn,只剩bimamba),46LA(45基础上删除下采样),48DF(同46模型)

#     # model = XLSX_ConMamba1_Model(args,device)#

#     # model = XLSX_SLS_Mamba_Model(args,device)#47LA,49DF，50LA，51LA,55LA(重跑47LA)

#     # model = XLSX_SLS_Dual_Mamba_Model(args,device)#52LA,53LA(+LayersNorm)

#     # model = XLSX_SLS_att_Mamba_Model(args,device)#54LA

#     # model = XLSX_SLS_Mamba2_Model(args,device)#52LA

#     # model = MambaModel(args,device)#57LA,58DF(ExBiMamba)

#     # model = XLSX_BiMamba_FFN_Model(args,device)#65LA,66DF(BiMamba_FFN),69DF(重跑66DF).72LA(再跑一次65LA)

#     # model = XLSX_BiMamba_FFN_Model(args,device)#75LA(再跑一次65LA),76DF((重跑66DF)BiMamba_FFN),MAMBA包不是pip的

#     model = XLSX_BiMamba_FFN_Model(args,device)#77LA(再跑一次65LA),78DF((重跑66DF)BiMamba_FFN)，11个mamba
#     #79LA(再跑一次65LA),80DF((重跑66DF)BiMamba_FFN)，10个mamba

#     # model = XLSX_BiMamba_FFN_SLS_Model(args,device)#70LA,71DF(BiMamba_FFN),

#     # model = ML_XLSX_BiMamba_FFN_SLS_Model(args,device)#73LA,74DF

#     # model = ML_XLSX_BiMamba_FFN_Model(args,device)#67LA,68DF(BiMamba_FFN)

#     # model = ML_MambaModel(args,device)#(SE)59LA,60DF

#     # model = ML_CBAM_MambaModel(args,device)#61LA,62DF

#     # model = OnlyMambaModel(args,device)#63LA,64DF(Mamba)

#     # model = InnerMambaModel(args,device)


#     # model = MambaHSI(device,in_channels=256, num_classes=2, hidden_dim=128)


#     # 定义模型保存与加载路径
#     model_save_path = os.path.join('./models', f'Exp{args.exp_num}')
#     best_save_path = os.path.join(model_save_path, 'best')
#     print('######## Eval ########')

#     # 加载训练好的模型权重
#     if args.average_model:
#         # 对多个最优模型求平均
#         model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0)), map_location=device))
#         print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
#         sd = model.state_dict()
#         for i in range(1, args.n_average_model):
#             model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i)), map_location=device))
#             print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
#             sd2 = model.state_dict()
#             for key in sd:
#                 sd[key] = (sd[key] + sd2[key])
#         for key in sd:
#             sd[key] = (sd[key]) / args.n_average_model
#         model.load_state_dict(sd)
#         print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
#     else:
#         # 加载单一best模型
#         model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth'), map_location=device))
#         print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

#     model.to(device)
#     model.eval()

#     # 开始计算RTF(重复200次求平均)
#     repeat_times = 200
#     rts = []   # 保存实时因子(RTF)
#     log_lines = []  # 保存日志信息的列表

#     for dur in utterance_durations:
#         waveform, sr = load_audio_segment(audio_file, dur)

#         total_time = 0.0
#         for _ in range(repeat_times):
#             start = time.time()
#             _ = model_inference(waveform, model, device)
#             end = time.time()
#             inference_time = end - start
#             total_time += inference_time

#         avg_time = total_time / repeat_times
#         audio_duration = dur
#         rtf = avg_time / audio_duration
#         rts.append(rtf)
        
#         # 打印并记录日志信息
#         log_str = f"Duration: {dur}s, Avg Processing time-inference_time (over {repeat_times} runs): {avg_time:.4f}s, RTF: {rtf:.4f}"
#         print(log_str)
#         log_lines.append(log_str)

#     # 可视化RTF与持续时间的关系
#     plt.figure(figsize=(6,4))
#     plt.plot(utterance_durations, rts, 'o-', color='red', label='Model RTF')
#     plt.xlabel('Utterance Duration (s)')
#     plt.ylabel('RTF')
#     plt.title('Inference RTF vs Utterance Duration (End-to-End Model)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()

#     # 设置y轴显示4位小数
#     plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
#     # 设置x轴为1到10的整数刻度
#     plt.xticks(utterance_durations)
    
#     # 将图表保存，以实验编号命名
#     fig_name = f"Exp{args.exp_num}_RTF.png"
#     fig_path = os.path.join(model_save_path, fig_name)
#     plt.savefig(fig_path)
#     print(f"RTF figure saved at {fig_path}")

#     # 将日志信息保存到txt文件
#     log_filename = f"Exp{args.exp_num}_RTF_log.txt"
#     log_filepath = os.path.join(model_save_path, log_filename)
#     with open(log_filepath, 'w') as f:
#         for line in log_lines:
#             f.write(line + '\n')
#     print(f"Log saved at {log_filepath}")

#     # 显示图表
#     plt.show()


import time
import torchaudio
import torch
import matplotlib.pyplot as plt
import argparse
import sys
import os
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_fixed_eval #Fixed
from model import *
import matplotlib.ticker as ticker
from thop import profile

def load_audio_segment(file_path, duration):
    """
    从完整音频中截取duration秒的音频片段
    假设音频为单声道，16kHz采样率（如实际数据不符请根据需要重采样）。
    """
    waveform, sr = torchaudio.load(file_path)
    target_length = int(sr * duration)
    
    if waveform.size(1) > target_length:
        # 截取前duration秒
        waveform = waveform[:, :target_length]
    else:
        # 如果音频短于所需长度，可以对其进行零填充
        pad_length = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    return waveform, sr

def model_inference(waveform, model, device):
    """
    模型推理函数：将端到端模型应用于输入波形数据。
    waveform形状假设为[channels, time]，根据你的模型需要可能进行变形。
    """
    # 假设模型需要 [batch, time] 且为单声道
    waveform_input = waveform.squeeze(0).unsqueeze(0).to(device)  # [1, time]
    
    # 推理
    with torch.no_grad():
        output = model(waveform_input)  # 假设 model(...) 直接输出结果
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    # Dataset
    parser.add_argument('--database_path', type=str, default='./datasets/cleaned/', help='Path to datasets')
    parser.add_argument('--protocols_path', type=str, default='./database/', help='Path to protocols')
    parser.add_argument('--exp_num', type=int, default=67, help='Experiment number for saving results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                        help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                        help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                        help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                        help='number of encoders of the conformer')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']))

    # model save path and evaluation parameters
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--comment_eval', type=str, default=None)
    parser.add_argument('--train', default='true', type=lambda x: (str(x).lower() in ['true', 'yes', '1']))
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']))
    parser.add_argument('--n_average_model', default=5, type=int)

    args = parser.parse_args()
    print(args)

    # 主程序：对不同秒长的语音计算RTF
    audio_file = "/home/xxuan/speech-deepfake/release_in_the_wild/1.wav"  # 请换成实际音频文件路径
    utterance_durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 修改后的语音片段时长（秒）

    #GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    # 初始化模型
    # model = XLSX_BiMamba_FFN_Model(args,device)#示例：根据需要修改模型类

    # 初始化模型
    # model = Model(args,device)#baseline,37DF(EXP9,16LA_FIXED)(EXP20DF_FIXED)

#     model = XLSR_BiMamba_CLS(args,device)#80LA,81DF

#     model = XLSR_BiMamba_CLS(args,device)#82LA,83DF-MyBiMamba-_get_clones(BiMambaEncoder

#     model = Model2(args,device)#baseline-val_test(EXP9,16LA_VAR)(EXP20DF_VAR)

#     model = TCM_Model(args,device)#TCM

#     model = ConBiMamba_Model(args,device)#exp15

#     model = Model(args,device)#baseline-exp16,conformer00

#     model = ConBiMamba_Model00(args,device)#exp17(conformer4,head4),18(conformer17,head4),17和18没有用MHSA，所以head没有传入
# #     # 19(FFN+BIMAMBA+MHSA+CONV+FFN,conformer17,head4),34,36,38DF
    

#     model = ConBiMamba_Model00_CLS(args,device)#exp23,24

#     model = ConBiMamba_Model00_CLS_ASP(args,device)#exp25,28

#     model = SSL_Model00_CLS_ASP(args,device)#exp26,27,35

#     model = ASP_ConBiMamba_Model00(args,device)#EXP

#     model = ConBiMamba_Model00_var(args,device)#exp18_test_val

#     model = ConBiMamba_AM_Model(args,device)

#     model = ConBiMamba_Model00_mutil_ASP(args,device)#EXP29,conformer_block=4

#     model = SSL_24output_ConBiMamba_Model00(args,device)#EXP32

#     model = (args,device)#exp

#     model = XLSX_SLS_Model(args,device)#40DF,42LA,43LA

#     model = XLSX_MLLA_Model(args,device)#LA

#     model = XLSX_CBAM_SLS_Model(args,device)#56LA

#     model = XLSX_Mamba1_Model(args,device)#41LA,44LA,45LA(conbimamba,删掉conv+ffn,只剩bimamba),46LA(45基础上删除下采样),48DF(同46模型)

#     model = XLSX_ConMamba1_Model(args,device)#

#     model = XLSX_SLS_Mamba_Model(args,device)#47LA,49DF，50LA，51LA,55LA(重跑47LA)

#     model = XLSX_SLS_Dual_Mamba_Model(args,device)#52LA,53LA(+LayersNorm)

#     model = XLSX_SLS_att_Mamba_Model(args,device)#54LA

#     model = XLSX_SLS_Mamba2_Model(args,device)#52LA

#     model = MambaModel(args,device)#57LA,58DF(ExBiMamba)

#     model = XLSX_BiMamba_FFN_Model(args,device)#65LA,66DF(BiMamba_FFN),69DF(重跑66DF).72LA(再跑一次65LA)

    # model = XLSX_BiMamba_FFN_Model(args,device)#75LA(再跑一次65LA),76DF((重跑66DF)BiMamba_FFN),MAMBA包不是pip的

    # model = XLSX_BiMamba_FFN_Model(args,device)#77LA(再跑一次65LA),78DF((重跑66DF)BiMamba_FFN)，11个mamba
# #     #79LA(再跑一次65LA),80DF((重跑66DF)BiMamba_FFN)，10个mamba

#     model = XLSX_BiMamba_FFN_SLS_Model(args,device)#70LA,71DF(BiMamba_FFN),

#     model = ML_XLSX_BiMamba_FFN_SLS_Model(args,device)#73LA,74DF

    model = ML_XLSX_BiMamba_FFN_Model(args,device)#67LA,68DF(BiMamba_FFN)

#     model = ML_MambaModel(args,device)#(SE)59LA,60DF

#     model = ML_CBAM_MambaModel(args,device)#61LA,62DF

#     model = OnlyMambaModel(args,device)#63LA,64DF(Mamba)

#     model = InnerMambaModel(args,device)


#     model = MambaHSI(device,in_channels=256, num_classes=2, hidden_dim=128)

    # 定义模型保存与加载路径
    model_save_path = os.path.join('./models', f'Exp{args.exp_num}')
    best_save_path = os.path.join(model_save_path, 'best')
    print('######## Eval ########')

    # 加载训练好的模型权重
    if args.average_model:
        # 对多个最优模型求平均
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_0.pth'), map_location=device))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_0.pth')))
        sd = model.state_dict()
        for i in range(1, args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, f'best_{i}.pth'), map_location=device))
            print('Model loaded : {}'.format(os.path.join(best_save_path, f'best_{i}.pth')))
            sd2 = model.state_dict()
            for key in sd:
                sd[key] = (sd[key] + sd2[key])
        for key in sd:
            sd[key] = (sd[key]) / args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth'), map_location=device))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

    model.to(device)
    model.eval()

    # 计算 FLOPs 和 Params
    # 构造一个dummy输入，例如1秒的语音数据（16kHz时16000点）
    dummy_input = torch.randn(1, 16000).to(device)
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    # 将 params 转为百万参数(M)方便阅读
    params_m = params / 1e6
    # 将flops转为GFLOPs方便阅读（如果数值较大）
    gflops = flops / 1e9

    print(f"Model FLOPs: {flops}, GFLOPs: {gflops:.4f}, Params: {params_m:.4f}M")

    # 开始计算RTF(重复200次求平均)
    repeat_times = 20
    rts = []   # 保存实时因子(RTF)
    log_lines = []  # 保存日志信息的列表

    # 记录FLOPs和Params信息到log中
    log_lines.append(f"Model FLOPs: {flops}, GFLOPs: {gflops:.4f}, Params: {params_m:.4f}M")

    for dur in utterance_durations:
        waveform, sr = load_audio_segment(audio_file, dur)
        waveform = waveform.to(device)  # [1, time]
        total_time = 0.0
        # 热身运行
        for _ in  range ( 10 ): 
            _ = model(waveform)
        # 多次迭代
        for _ in range(repeat_times):
            torch.cuda.synchronize()  # 同步保证计时准确
            start = time.time()        
            _ = model_inference(waveform, model, device)
            torch.cuda.synchronize()  # 再次同步确保所有GPU操作完成后再计时
            end = time.time()
            inference_time = end - start
            total_time += inference_time

        avg_time = total_time / repeat_times
        audio_duration = dur
        rtf = avg_time / audio_duration
        rts.append(rtf)
        
        # 打印并记录日志信息
        log_str = f"Duration: {dur}s, Avg Processing time (over {repeat_times} runs): {avg_time:.4f}s, RTF: {rtf:.4f}"
        print(log_str)
        log_lines.append(log_str)

    # 可视化RTF与持续时间的关系
    plt.figure(figsize=(6,4))
    plt.plot(utterance_durations, rts, 'o-', color='red', label='Model RTF')
    plt.xlabel('Utterance Duration (s)')
    plt.ylabel('RTF')
    plt.title('Inference RTF vs Utterance Duration (End-to-End Model)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 设置y轴显示4位小数
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    # 设置x轴为1到10的整数刻度
    plt.xticks(utterance_durations)
    
    # 将图表保存，以实验编号命名
    fig_name = f"Exp{args.exp_num}_RTF_warmup.png"
    # fig_name = f"Exp{args.exp_num}_RTF_no_warmup.png"
    fig_path = os.path.join(model_save_path, fig_name)
    plt.savefig(fig_path)
    print(f"RTF figure saved at {fig_path}")

    # 将日志信息保存到txt文件
    log_filename = f"Exp{args.exp_num}_RTF_log_warmup.txt"
    # log_filename = f"Exp{args.exp_num}_RTF_log_no_warmup.txt"
    log_filepath = os.path.join(model_save_path, log_filename)
    with open(log_filepath, 'w') as f:
        for line in log_lines:
            f.write(line + '\n')
    print(f"Log saved at {log_filepath}")

    # 显示图表
    plt.show()

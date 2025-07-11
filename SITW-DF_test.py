import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_ASVSpoof19_eval
from model import *
from utils import reproducibility
from utils import read_metadata,compute_eer
import numpy as np

import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# from dataset import ASVspoof2019
# from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from evaluate_tDCF_asvspoof19 import calculate_tDCF_EER,calculate_EER
from tqdm import tqdm
import eval_metrics as em
import numpy as np
from ASVspoof2019 import ASVspoof2019,SITW,SITW_all,SITW_Var



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    # Dataset
    parser.add_argument('--database_path', type=str, default='./datasets/cleaned/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/wav
    %      |- ASVspoof2019_LA_train/wav
    %      |- ASspoof2019_LA_dev/wav
    %      |- ASVspoof2021_DF_eval/wav
    '''

    parser.add_argument('--protocols_path', type=str, default='./database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--exp_num', type=int, default=1, help='Experiment number for saving results')
    parser.add_argument('--batch_size', type=int, default=16)
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
    # parser.add_argument('--num_encoders', type=int, default=12, metavar='N',
    #                 help='number of encoders of the conformer')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    
    #Train
    parser.add_argument('--train', default='true', type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)

    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    print(args)
    args.track='LA'
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores=args.n_mejores_loss

    assert track in ['LA','DF'], 'Invalid track given'
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'Conformer_{}_{}_{}_ES{}_H{}_NE{}_KS{}'.format(
        track, args.loss, args.lr,args.emb_size, args.heads, args.num_encoders, args.kernel_size)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    # model_save_path = os.path.join('models', model_tag)
    
    # model_save_path = os.path.join('./store_models', 'Exp16')############################
    # model_save_path = os.path.join('./models', 'Exp65')############################
    model_save_path = os.path.join('/nlp/nlp-xxuan/models/', 'Exp174')############################/.'  yt67
    
    # print('Model tag: '+ model_tag)s
    print("model_save_path",model_save_path)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    #GPU device
    import torch

    # # 设置当前的 GPU 设备为 cuda:1
    # torch.cuda.set_device(1)

    # # 清理 cuda:1 的缓存
    # torch.cuda.empty_cache()



    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    torch.cuda.empty_cache()
    
    # model = Model(args,device)#9

    # model = XLSR_BiMamba_CLS(args,device)#80LA

    # model = XLSR_BiMamba_CLS(args,device)#82LA,83DF-MyBiMamba-_get_clones(BiMambaEncoder

    # model = Model(args,device)#baseline,37DF(EXP9,16LA_FIXED)(EXP20DF_FIXED)

    # model = Model2(args,device)#baseline-val_test(EXP9,16LA_VAR)(EXP20DF_VAR)

    # model = ConBiMamba_Model(args,device)

    # model = ConBiMamba_Model00(args,device)#exp18,22 #7要加上fc

    # model = ConBiMamba_Model00_CLS(args,device)#exp23,24

    # model = ConBiMamba_Model00_CLS_ASP(args,device)#exp25

    # model = SSL_Model00_CLS_ASP(args,device)#exp26

    # model = ConBiMamba_Model00_CLS_ASP(args,device)#exp25,28
    #----------------------------------------------------------------
    # model = Model(args,device)#baseline,37DF

    # model = Model2(args,device)#baseline-val_test,39

    # model = TCM_Model(args,device)#TCM

    # model = ConBiMamba_Model(args,device)#exp15

    # model = Model(args,device)#baseline-exp16,conformer00

    # model = ConBiMamba_Model00(args,device)#exp17(conformer4,head4),18(conformer17,head4),17和18没有用MHSA，所以head没有传入
    # 19(FFN+BIMAMBA+MHSA+CONV+FFN,conformer17,head4),34,36,38DF
    

    # model = ConBiMamba_Model00_CLS(args,device)#exp23,24

    # model = ConBiMamba_Model00_CLS_ASP(args,device)#exp25,28

    # model = SSL_Model00_CLS_ASP(args,device)#exp26,27,35

    # model = ASP_ConBiMamba_Model00(args,device)#EXP

    # model = ConBiMamba_Model00_var(args,device)#exp18_test_val

    # model = ConBiMamba_AM_Model(args,device)

    # model = ConBiMamba_Model00_mutil_ASP(args,device)#EXP29,conformer_block=4

    # model = SSL_24output_ConBiMamba_Model00(args,device)#EXP32

    # model = XLSX_SLS_Model(args,device)#40DF,42LA
    

    # model = XLSX_Mamba1_Model(args,device)#41LA,44LA,45LA(conbimamba,删掉conv+ffn,只剩bimamba),46LA(45基础上删除下采样)

    # model = XLSX_SLS_Mamba_Model(args,device)#47LA

    # model = XLSX_SLS_Mamba_Model(args,device)#47LA,49DF，50LA,51LA

    # model = XLSX_SLS_Mamba_Model(args,device)#47LA,49DF，50LA，51LA,55LA(重跑47LA)

    # model = XLSX_SLS_Dual_Mamba_Model(args,device)#52LA,53LA(+LayersNorm)

    # model = XLSX_CBAM_SLS_Model(args,device)#56LA

    # model = MambaModel(args,device)#57LA,58DF

    # model = ML_MambaModel(args,device)#(SE)59LA,60DF

    # model = ML_CBAM_MambaModel(args,device)#61LA,62DF

    # model = OnlyMambaModel(args,device)#63LA,64DF(Mamba)

    # model = XLSX_BiMamba_FFN_Model(args,device)#65LA,66DF(BiMamba_FFN)

    # model = ML_XLSX_BiMamba_FFN_Model(args,device)#67LA,68DF(BiMamba_FFN)

    # model = ML_XLSX_10_BiMamba_FFN_Model(args,device)

    # model = ML_XLSX_BiMamba_FFN_SLS_Model(args,device)#73LA,74DF

    # model = ML_XLSR_Conformer(args,device)

    model = XLSR_Conformer(args,device)#和XLSR_BIMAMBA对比，改掉encoder，exp132


    # model = XLSR_AttW_Conformer(args,device)#和XLSR_BIMAMBA对比，改掉encoder，exp142

    # model = XLSX_Transformer(args,device)




    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
     
    # define train dataloader
    label_trn, files_id_train = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of training trials',len(files_id_train))
    
    # train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 32, shuffle=True, drop_last = True)
    
    # del train_set, label_trn
    
    # # define validation dataloader
    # labels_dev, files_id_dev = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)), is_eval=False)
    # print('no. of validation trials',len(files_id_dev))

    # dev_set = Dataset_train(args,list_IDs = files_id_dev,
	# 	    labels = labels_dev,
	# 	    base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)), algo=args.algo)

    # dev_loader = DataLoader(dev_set, batch_size=10, num_workers=32, shuffle=False)
    # del dev_set,labels_dev

    print(f"Current device: {torch.cuda.current_device()}")




    print('######## Eval ########')
    if args.average_model:
        sdl=[]
        print("best_save_path",best_save_path)
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        sd = model.state_dict()
        for i in range(1,args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key]=(sd[key]+sd2[key])
        for key in sd:
            sd[key]=(sd[key])/args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))


    #加载一个模型，best.pth
    # model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
    # print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # # #--------------------------------------T-SNE-SITW----------------------------------------------

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.manifold import TSNE
    from sklearn.utils import shuffle

    # os.environ['OPENBLAS_NUM_THREADS'] = '1'


    # 确保目录存在
    best_save_path = os.path.join(model_save_path, 'best', 'SITW', 'score.txt')
    # best_save_path = os.path.join(model_save_path, 'best', 'SITW', 'best-score.txt')
    best_save_dir = os.path.dirname(best_save_path)

    if not os.path.exists(best_save_dir):
        os.makedirs(best_save_dir)
        print(f"Created directory: {best_save_dir}")

    # 加载测试数据
    test_set = SITW_all("LA", 
                        "/home/xxuan/speech-deepfake/DF-SITW/SITW-labels.txt", 
                        padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10,
                                collate_fn=test_set.collate_fn)
    model.eval()

    # 写入文件并提取嵌入
    cm_score_path = best_save_path
    embeddings_list = []
    labels_list = []
    audio_filenames = []

    with open(cm_score_path, 'w') as cm_score_file:
        for i, (batch_x, audio_fn, labels) in enumerate(tqdm(testDataLoader)):
            batch_x = batch_x.to(device)
            labels = labels.to(device)

            # 获取分类分数和嵌入（假设模型返回嵌入）
            out, embedding = model(batch_x, return_embedding=True)  # 修改为支持返回嵌入
            score = out[:, 1].data.cpu().numpy().ravel()

            # 保存分数到文件
            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s %s %.6f\n' % (audio_fn[j], labels[j].data.cpu().numpy(), score[j])
                )

            # 保存嵌入、标签和音频文件名
            embeddings_list.append(embedding.data.cpu().numpy())
            labels_list.extend(labels.data.cpu().numpy())
            audio_filenames.extend(audio_fn)

    # 合并所有嵌入
    embeddings = np.concatenate(embeddings_list, axis=0)  # (num_samples, embedding_dim)
    labels = np.array(labels_list)  # (num_samples,)

    # 随机采样（限制最大样本数）
    max_samples = 10000
    if embeddings.shape[0] > max_samples:
        embeddings, labels = shuffle(embeddings, labels, random_state=42)
        embeddings = embeddings[:max_samples]
        labels = labels[:max_samples]

    # 读取 CM scores
    cm_data = np.genfromtxt(cm_score_path, dtype=str)
    cm_keys = cm_data[:, 1]  # 确保是字符串数组
    cm_scores = cm_data[:, 2].astype(float)  # 修改为内置 float，避免 NumPy 警告

    # 提取 bona fide 和 spoof 分数
    bona_cm = cm_scores[cm_keys == '0']  # 假设 bonafide 为 '0'
    spoof_cm = cm_scores[cm_keys == '1']  # 假设 spoof 为 '1'

    # 异常检查
    if len(bona_cm) == 0 or len(spoof_cm) == 0:
        raise ValueError("无法提取 bona fide 或 spoof 分数，请检查数据！")

    # 计算 EER
    eer_cm_result = compute_eer(bona_cm, spoof_cm)[0]
    print(f"Pooled EER CM Result: {eer_cm_result}")

    # t-SNE 降维
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)

    # 设置 t-SNE 参数
    # tsne = TSNE(
    #     n_components=2,       # 降维到二维
    #     perplexity=50,        # 增加 perplexity 以捕捉全局结构
    #     learning_rate=500,    # 提高学习率以加快收敛
    #     n_iter=2000,          # 增加迭代次数
    #     random_state=42       # 保持结果可复现
    # )
    
    reduced_embeddings = tsne.fit_transform(embeddings)

    # # 绘制 t-SNE 可视化
    # print("Generating t-SNE visualization...")
    # plt.figure(figsize=(8, 8))
    # for label, color, name in zip([0, 1], ['cornflowerblue', 'lightcoral'], ['Bonafide', 'Spoof']):
    #     plt.scatter(
    #         reduced_embeddings[labels == label, 0],
    #         reduced_embeddings[labels == label, 1],
    #         c=color, label=name, s=5
    #     )
    # plt.legend()
    # # plt.title('t-SNE Visualization of before LAP')
    # # tsne_image_path = os.path.join(best_save_dir, 't-SNE+before_LAP+visualization.png')
    # plt.title('t-SNE Visualization of LAP later')
    # tsne_image_path = os.path.join(best_save_dir, '1w_t-SNE+LAP_later+visualization.png')
    # plt.savefig(tsne_image_path)
    # plt.show()

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from matplotlib import font_manager

    


    # 多种配色方案
    # color_schemes = [
    #     {0: ('cornflowerblue', 'Real'), 1: ('lightcoral', 'Fake')},
    #     {0: ('#BEBAB9', 'Real'), 1: ('#C47070', 'Fake')},
    #     {0: ('#9BBBE1', 'Real'), 1: ('#F09BA0', 'Fake')},
    #     {0: ('#BEBAB9', 'Real'), 1: ('#BE7FB8', 'Fake')}
    # ]
    color_schemes = [
    # # 紫色配色
    # {0: ('#D9B4D5', 'Real'), 1: ('#C367A2', 'Fake')},

    # # 蓝色配色
    # {0: ('#A8D0E6', 'Real'), 1: ('#4B7FB8', 'Fake')},

    # # 红色配色
    # {0: ('#F6B7B0', 'Real'), 1: ('#D76C77', 'Fake')},

    # # 绿色配色
    # {0: ('#B9E3B6', 'Real'), 1: ('#6E9D70', 'Fake')},

    # # 黄色配色
    # {0: ('#F9EBA7', 'Real'), 1: ('#D0A24C', 'Fake')},

    # # 橙色配色
    # {0: ('#F6D3B2', 'Real'), 1: ('#F37D3F', 'Fake')},

    # # 粉色配色
    # {0: ('#F1C1D6', 'Real'), 1: ('#E66A8F', 'Fake')},

    # 深R蓝-F红
    {0: ('#4B7FB8', 'Real'), 1: ('#D76C77', 'Fake')},
    
    # 深R蓝-F紫
    {0: ('#4B7FB8', 'Real'), 1: ('#C367A2', 'Fake')},

    # 浅R蓝-F红
    {0: ('#A8D0E6', 'Real'), 1: ('#F6B7B0', 'Fake')},
    
    # 浅R蓝-F紫
    {0: ('#A8D0E6', 'Real'), 1: ('#D9B4D5', 'Fake')}
]


    # 计算最近邻（k-NN）混淆样本点
    k = 5  # 取每个点的 5 个最近邻
    nbrs = NearestNeighbors(n_neighbors=k).fit(reduced_embeddings)
    distances, indices = nbrs.kneighbors(reduced_embeddings)

    # 计算混淆点数量
    confused_samples = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = labels[neighbors]  # 取最近邻的label
        majority_label = np.bincount(neighbor_labels).argmax()  # 统计多数类别
        if majority_label != labels[i]:  # 如果大多数邻居的类别与当前点不同，认为是混淆样本
            confused_samples += 1

    print(f"混淆样本点的个数: {confused_samples}")

    # 生成并保存不同配色的 t-SNE 可视化
    for i, label_colors in enumerate(color_schemes):
        print(f"Generating t-SNE visualization for color scheme {i + 1}...")

        plt.figure(figsize=(8, 8), dpi=300)  # 高分辨率图片
        for label, (color, name) in label_colors.items():
            plt.scatter(
                reduced_embeddings[labels == label, 0],
                reduced_embeddings[labels == label, 1],
                c=color, label=name, s=10,  # s 控制点大小
                # edgecolors='black', linewidths=1.2  # 黑色边框，边框宽度1.2
            )

        plt.legend(fontsize=12)
        plt.title(f't-SNE Visualization (Confused Samples: {confused_samples})', fontsize=14)
        plt.xticks([])  # 去掉横轴刻度
        plt.yticks([])  # 去掉纵轴刻度

        # 生成文件名
        color_names = "_".join([color.replace('#', '') for color, _ in label_colors.values()])
        tsne_image_path = os.path.join(best_save_dir, f'tSNE_{color_names}.png')

        # 保存高清图片
        plt.savefig(tsne_image_path, bbox_inches='tight', dpi=300)
        plt.show()

        print(f"t-SNE visualization saved to {tsne_image_path}")


    #-----------------------------ALL-------------------------------------------------------------
    # import os
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from tqdm import tqdm
    # from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE
    # from sklearn.utils import shuffle

    # # 确保目录存在
    # best_save_path = os.path.join(model_save_path, 'best', 'SITW', 'score.txt')
    # best_save_dir = os.path.dirname(best_save_path)

    # if not os.path.exists(best_save_dir):
    #     os.makedirs(best_save_dir)
    #     print(f"Created directory: {best_save_dir}")

    # # 加载测试数据
    # test_set = SITW_all("LA", 
    #                     "/home/xxuan/speech-deepfake/DF-SITW/SITW-labels.txt", 
    #                     padding="repeat")
    # testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10,
    #                             collate_fn=test_set.collate_fn)
    # model.eval()

    # # 写入文件并提取嵌入
    # cm_score_path = best_save_path
    # embeddings_list = []
    # labels_list = []
    # audio_filenames = []

    # with open(cm_score_path, 'w') as cm_score_file:
    #     for i, (batch_x, audio_fn, labels) in enumerate(tqdm(testDataLoader)):
    #         batch_x = batch_x.to(device)
    #         labels = labels.to(device)

    #         # 获取分类分数和嵌入（假设模型返回嵌入）
    #         out, embedding = model(batch_x, return_embedding=True)
    #         score = out[:, 1].data.cpu().numpy().ravel()

    #         # 保存分数到文件
    #         for j in range(labels.size(0)):
    #             cm_score_file.write(
    #                 '%s %s %.6f\n' % (audio_fn[j], labels[j].data.cpu().numpy(), score[j])
    #             )

    #         # 保存嵌入、标签和音频文件名
    #         embeddings_list.append(embedding.data.cpu().numpy())
    #         labels_list.extend(labels.data.cpu().numpy())
    #         audio_filenames.extend(audio_fn)

    # # 合并所有嵌入
    # embeddings = np.concatenate(embeddings_list, axis=0)
    # labels = np.array(labels_list)

    # # 读取 CM scores
    # cm_data = np.genfromtxt(cm_score_path, dtype=str)
    # cm_keys = cm_data[:, 1]  # 确保是字符串数组
    # cm_scores = cm_data[:, 2].astype(float)  # 修改为内置 float，避免 NumPy 警告

    # # 提取 bona fide 和 spoof 分数
    # bona_cm = cm_scores[cm_keys == '0']  # 假设 bonafide 为 '0'
    # spoof_cm = cm_scores[cm_keys == '1']  # 假设 spoof 为 '1'

    # # 统计 bonafide 和 spoof 的个数
    # bona_count = len(bona_cm)
    # spoof_count = len(spoof_cm)

    # print(f"Bonafide Count: {bona_count}, Spoof Count: {spoof_count}")

    # # 异常检查
    # if len(bona_cm) == 0 or len(spoof_cm) == 0:
    #     raise ValueError("无法提取 bona fide 或 spoof 分数，请检查数据！")

    # # 计算 EER
    # eer_cm_result = compute_eer(bona_cm, spoof_cm)[0]
    # print(f"Pooled EER CM Result: {eer_cm_result}")

    # # 使用 PCA 将维度降到 intermediate_dim
    # intermediate_dim = 50
    # print("Reducing dimensions using PCA...")
    # pca = PCA(n_components=intermediate_dim)
    # embeddings_pca = pca.fit_transform(embeddings)

    # # 分批运行 t-SNE
    # def batch_tsne(embeddings, batch_size=5000, tsne_params=None):
    #     if tsne_params is None:
    #         tsne_params = {
    #             "n_components": 2,
    #             "perplexity": 30,
    #             "learning_rate": 200,
    #             "n_iter": 1000,
    #             "random_state": 42,
    #         }

    #     reduced_embeddings = []
    #     num_batches = (embeddings.shape[0] + batch_size - 1) // batch_size

    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = min((i + 1) * batch_size, embeddings.shape[0])
    #         print(f"Processing batch {i + 1}/{num_batches}, samples {start_idx}:{end_idx}...")
            
    #         tsne = TSNE(**tsne_params)
    #         batch_reduced = tsne.fit_transform(embeddings[start_idx:end_idx])
    #         reduced_embeddings.append(batch_reduced)

    #     return np.vstack(reduced_embeddings)

    # print("Performing t-SNE in batches...")
    # reduced_embeddings = batch_tsne(embeddings_pca, batch_size=5000)

    # # 绘制 t-SNE 可视化
    # print("Generating t-SNE visualization...")
    # plt.figure(figsize=(8, 8))
    # for label, color, name in zip([0, 1], ['cornflowerblue', 'lightcoral'], ['Bonafide', 'Spoof']):
    #     plt.scatter(
    #         reduced_embeddings[labels == label, 0],
    #         reduced_embeddings[labels == label, 1],
    #         c=color, label=name, s=5
    #     )
    # plt.legend()
    # plt.title('t-SNE Visualization of LAP later')
    # tsne_image_path = os.path.join(best_save_dir, 'All_samples_t-SNE_LAP_later_visualization.png')
    # plt.savefig(tsne_image_path)
    # plt.show()

    # print(f"t-SNE visualization saved to {tsne_image_path}")



    #---------------------------------------------------------------------------------------------
    # import os
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from tqdm import tqdm
    # from sklearn.utils import shuffle

    # # 确保目录存在
    # best_save_path = os.path.join(model_save_path, 'best', 'SITW', 'score.txt')
    # best_save_dir = os.path.dirname(best_save_path)

    # if not os.path.exists(best_save_dir):
    #     os.makedirs(best_save_dir)
    #     print(f"Created directory: {best_save_dir}")

    # # 加载测试数据
    # test_set = SITW_all(
    #     "LA", 
    #     "/home/xxuan/speech-deepfake/DF-SITW/SITW-labels.txt", 
    #     padding="repeat"
    # )
    # testDataLoader = DataLoader(
    #     test_set, batch_size=1, shuffle=False, num_workers=10,
    #     collate_fn=test_set.collate_fn
    # )
    # model.eval()

    # # 提取分数和标签
    # scores_list = []
    # labels_list = []
    # audio_filenames = []

    # with open(best_save_path, 'w') as cm_score_file:
    #     for i, (batch_x, audio_fn, labels) in enumerate(tqdm(testDataLoader)):
    #         batch_x = batch_x.to(device)
    #         labels = labels.to(device)

    #         # 获取分类分数
    #         out = model(batch_x)  # 模型直接输出分类分数
    #         scores = out.data.cpu().numpy()  # 假设 out 是二分类概率 (batch_size, 2)

    #         # 保存分数到文件
    #         for j in range(labels.size(0)):
    #             cm_score_file.write(
    #                 '%s %s %.6f %.6f\n' % (audio_fn[j], labels[j].data.cpu().numpy(), scores[j][0], scores[j][1])
    #             )

    #         # 保存分数和标签
    #         scores_list.append(scores)
    #         labels_list.extend(labels.data.cpu().numpy())
    #         audio_filenames.extend(audio_fn)

    # # 合并所有分数和标签
    # scores = np.vstack(scores_list)  # (num_samples, 2)
    # labels = np.array(labels_list)  # (num_samples,)

    # # 可视化分布
    # plt.figure(figsize=(8, 8))
    # for label, color, name in zip([0, 1], ['cornflowerblue', 'lightcoral'], ['Bonafide', 'Spoof']):
    #     plt.scatter(
    #         scores[labels == label, 0],  # 类别为 label 的样本的第一个分数
    #         scores[labels == label, 1],  # 类别为 label 的样本的第二个分数
    #         c=color, label=name, s=10, alpha=0.7
    #     )
    # plt.xlabel('Score for Class 0 (Bonafide)')
    # plt.ylabel('Score for Class 1 (Spoof)')
    # plt.title('Score Distribution for Two Classes')
    # plt.legend()
    # plt.grid(True)

    # # 保存图像
    # score_dist_image_path = os.path.join(best_save_dir, 'score_distribution.png')
    # plt.savefig(score_dist_image_path)
    # plt.show()

    # print(f"Score distribution visualization saved to {score_dist_image_path}")




    # #--------------------------------------ALL-SITW----------------------------------------------

    # best_save_path = os.path.join(model_save_path, 'best', 'SITW', 'score.txt')
    # # best_save_path = os.path.join(model_save_path, 'best', 'SITW', 'score-var.txt')
    # best_save_dir = os.path.dirname(best_save_path)

    # # 确保目录存在
    # if not os.path.exists(best_save_dir):
    #     os.makedirs(best_save_dir)
    #     print(f"Created directory: {best_save_dir}")


    # # 加载测试数据
    # test_set = SITW_all("LA", 
    #                 "/home/xxuan/speech-deepfake/DF-SITW/SITW-labels.txt", 
    #                 padding="repeat")
    # # test_set = SITW_Var("LA", 
    # #                 "/home/xxuan/speech-deepfake/DF-SITW/SITW-labels.txt", 
    # #                 padding="repeat")
    # testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10,
    #                             collate_fn=test_set.collate_fn)
    # model.eval()
    

    # # 写入文件
    # cm_score_path = best_save_path
    # with open(cm_score_path, 'w') as cm_score_file:
    #     for i, (batch_x, audio_fn, labels) in enumerate(tqdm(testDataLoader)):
    #         # print("labels",labels)
    #         # print("labels",labels.shape)
    #         # print("labels",labels[0].shape)
    #         batch_x = batch_x.to(device)
    #         labels = labels.to(device)
    #         # labels = labels[0].to(device)
    #         out,emb = model(batch_x, return_embedding=True)
    #         score = out[:, 1].data.cpu().numpy().ravel()

    #         for j in range(labels.size(0)):
    #             cm_score_file.write(
    #                 '%s %s %.6f\n' % (audio_fn[j], labels[j].data.cpu().numpy(), score[j])
    #             )

    # # 读取 CM scores
    # cm_data = np.genfromtxt(cm_score_path, dtype=str)
    # cm_keys = cm_data[:, 1]  # 确保是字符串数组
    # cm_scores = cm_data[:, 2].astype(np.float)

    # # 提取 bona fide 和 spoof 分数
    # bona_cm = cm_scores[cm_keys == '0']  # 假设 bonafide 为 '0'
    # spoof_cm = cm_scores[cm_keys == '1']  # 假设 spoof 为 '1'

    # # 异常检查
    # if len(bona_cm) == 0 or len(spoof_cm) == 0:
    #     raise ValueError("无法提取 bona fide 或 spoof 分数，请检查数据！")

    # # 计算 EER
    # eer_cm_result = compute_eer(bona_cm, spoof_cm)[0]

    # # 打印结果
    # print(f"Pooled EER CM Result: {eer_cm_result}")

    #-------------------------------diff-utt---------------------------------------

    # 子集标签文件列表
    # label_files = [
    #     "SITW-labels->6s.txt",
    #     "SITW-labels-0-1s.txt",
    #     "SITW-labels-1-2s.txt",
    #     "SITW-labels-2-3s.txt",
    #     "SITW-labels-3-4s.txt",
    #     "SITW-labels-4-5s.txt",
    #     "SITW-labels-5-6s.txt"
    # ]

    label_files = [
        "SITW-labels-<3s.txt",
        "SITW-labels-3-4s.txt",
        "SITW-labels-4-5s.txt",
        "SITW-labels-5-6s.txt",
        "SITW-labels->6s.txt"
    ]

    
    # 保存结果的字典
    eer_results = {}


    # 遍历每个子集
    for label_file in label_files:
        print(f"Processing subset: {label_file}")

        # 构造保存路径
        cm_score_path = os.path.join(model_save_path, 'best', 'SITW', f"{label_file.split('.')[0]}_score.txt")
        best_save_dir = os.path.dirname(cm_score_path)
        
        # 确保目录存在
        if not os.path.exists(best_save_dir):
            os.makedirs(best_save_dir)
            print(f"Created directory: {best_save_dir}")
        
        # 加载测试数据
        test_set = SITW(
            os.path.join("/home/xxuan/speech-deepfake/DF-SITW/diff", label_file), 
            padding="repeat"
        )
        testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10, collate_fn=test_set.collate_fn)

        # 写入分数到文件
        with open(cm_score_path, 'w') as cm_score_file:
            for i, (batch_x, audio_fn, labels) in enumerate(tqdm(testDataLoader)):
                batch_x = batch_x.to(device)
                labels = labels.to(device)
                out = model(batch_x)
                score = out[:, 1].data.cpu().numpy().ravel()

                for j in range(labels.size(0)):
                    cm_score_file.write(
                        '%s %s %.6f\n' % (audio_fn[j], labels[j].data.cpu().numpy(), score[j])
                    )

        # 读取 CM scores
        cm_data = np.genfromtxt(cm_score_path, dtype=str)
        cm_keys = cm_data[:, 1]  # 确保是字符串数组
        cm_scores = cm_data[:, 2].astype(np.float)

        # 提取 bona fide 和 spoof 分数
        bona_cm = cm_scores[cm_keys == '0']  # 假设 bona fide 为 '0'
        spoof_cm = cm_scores[cm_keys == '1']  # 假设 spoof 为 '1'

        # 异常检查
        if len(bona_cm) == 0 or len(spoof_cm) == 0:
            raise ValueError(f"无法提取 bona fide 或 spoof 分数，请检查数据！Subset: {label_file}")

        # 计算 EER
        eer_cm_result = compute_eer(bona_cm, spoof_cm)[0]

        # 保存结果
        eer_results[label_file] = eer_cm_result

        # 打印子集结果
        print(f"Subset: {label_file}, EER: {eer_cm_result}")

    # 打印所有子集的结果
    print("\nFinal EER Results:")
    for subset, eer in eer_results.items():
        print(f"{subset}: {eer}")

    # 将结果保存到文件
    result_file_path = os.path.join(model_save_path, "itw-diff_sec_eer_results00.txt")
    with open(result_file_path, "w") as result_file:
        for subset, eer in eer_results.items():
            result_file.write(f"{subset}: {eer:.6f}\n")
    print(f"EER results saved to {result_file_path}")




import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_eval
from model import Model
from utils import reproducibility
from utils import read_metadata
import numpy as np

# 用于保存训练和验证的损失值和准确率到文件的函数
def save_results(exp_num, epoch, train_loss, train_acc, val_loss, val_acc):
    result_path = f'./results/exp{exp_num}/result.txt'
    
    # 如果文件路径不存在，则创建它
    if not os.path.exists(f'./results/exp{exp_num}'):
        os.makedirs(f'./results/exp{exp_num}')
    
    # 将结果写入文件
    with open(result_path, 'a') as f:
        f.write(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
    print(f"Results saved to {result_path}")

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    i = 0

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            # 将数据移至设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # 模型前向传播
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size
            
            # 计算预测结果
            pred = batch_out.max(1)[1]  # 获取预测值
            correct += pred.eq(batch_y).sum().item()  # 统计预测正确的数量
            
            # 计算当前批次的准确率
            batch_accuracy = (pred == batch_y).sum().item() / batch_size
            
            i += 1

            # 输出损失和准确率
            print("Batch %i of %i - Loss: %.4f, Accuracy: %.2f%% (Memory: %.2f of %.2f GiB reserved)" 
                  % (
                     i,
                     num_batch,
                     batch_loss.item(),
                     batch_accuracy * 100,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )
    
    # 计算整个验证集的平均损失和准确率
    val_loss /= num_total
    test_accuracy = 100. * correct / num_total
    print('\nTest accuracy: {:.2f}%'.format(test_accuracy))

    return val_loss, test_accuracy


from tqdm import tqdm  # 导入tqdm用于可视化进度条

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    total_samples = len(dataset)  # 获取总样本数
    
    fname_list = []
    score_list = []

    # 使用tqdm包裹data_loader以显示进度
    with tqdm(total=total_samples, desc="Processing Samples", unit="samples") as pbar:
        for batch_x, utt_id in data_loader:
            fname_list = []
            score_list = []
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
            
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            # 保存结果到文件
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list, score_list):
                    fh.write('{} {}\n'.format(f, cm))
            
            # 更新进度条，累加处理过的样本数
            pbar.update(len(batch_x))  # len(batch_x) 是本次批次的样本数量

    print('Scores saved to {}'.format(save_path))


import torch
import torch.nn as nn
import torch.optim as optim
import sys

def train_epoch(train_loader, model, lr, optimizer, device):
    num_total = 0.0
    num_correct = 0.0  # 用于统计正确预测的数量
    total_loss = 0.0  # 用于累加每个批次的损失
    model.train()

    # 设置损失函数
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i = 0

    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        # 将数据移至设备
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # 模型前向传播
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)     
        
        # 计算预测结果
        _, predicted = torch.max(batch_out, 1)  # 获取预测值
        num_correct += (predicted == batch_y).sum().item()  # 统计预测正确的数量

        # 优化器操作
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # 计算当前批次的准确率
        batch_accuracy = (predicted == batch_y).sum().item() / batch_size
        
        i += 1
        
        # 输出损失和准确率
        print("Batch %i of %i - Loss: %.4f, Accuracy: %.2f%% (Memory: %.2f of %.2f GiB reserved)" 
              % (
                 i,
                 num_batch,
                 batch_loss.item(),
                 batch_accuracy * 100,
                 torch.cuda.max_memory_allocated(device) / (2 ** 30),
                 torch.cuda.max_memory_reserved(device) / (2 ** 30),
                 ),
              end="\r",
              )

    # 计算整个 epoch 的平均损失和准确率
    epoch_loss = total_loss / num_total
    epoch_accuracy = num_correct / num_total * 100
    print("\nEpoch Loss: %.4f, Accuracy: %.2f%%" % (epoch_loss, epoch_accuracy))
    sys.stdout.flush()

    return epoch_loss, epoch_accuracy


    

       

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

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
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
    
    model_save_path = os.path.join('./store_models', 'Conformer_LA_WCE_1e-06_ES144_H4_NE4_KS31')############################
    
    # print('Model tag: '+ model_tag)
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

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
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
    
    train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 32, shuffle=True, drop_last = True)
    
    del train_set, label_trn
    
    # define validation dataloader
    labels_dev, files_id_dev = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of validation trials',len(files_id_dev))

    dev_set = Dataset_train(args,list_IDs = files_id_dev,
		    labels = labels_dev,
		    base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)), algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=10, num_workers=32, shuffle=False)
    del dev_set,labels_dev




    print('######## Eval ########')
    if args.average_model:
        sdl=[]
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

    eval_tracks=['DF']#,'LA'
    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)

    for tracks in eval_tracks:
        print("tracks", tracks)
        best_save_path = os.path.join(model_save_path, 'best', 'DF21', 'score.txt')

        # Check if the directory of best_save_path exists, if not, create it
        best_save_dir = os.path.dirname(best_save_path)

        if not os.path.exists(best_save_dir):
            os.makedirs(best_save_dir)  # Create the directories if they do not exist
            print(f"Created directory: {best_save_dir}")

        if not os.path.exists(best_save_path):
            prefix      = 'ASVspoof_{}'.format(tracks)
            prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
            prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

            file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)), is_eval=True)
            print('no. of eval trials',len(file_eval))
            eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(tracks)),track=tracks)
            produce_evaluation_file(eval_set, model, device, best_save_path)
        else:
            print('Score file already exists')

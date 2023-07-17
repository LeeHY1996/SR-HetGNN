
import torch
import numpy as np
import argparse
import pickle
import time
import os
from utils import  Data, split_validation ,het_neigh,get_embedding
from model import *


#参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='short_diginetica', help='dataset name: diginetica/short_diginetica/yoochoose1_64/sample')
parser.add_argument('--data_path', type = str, default = '../datasets/Tmall/',help='path to data')
parser.add_argument('--u_N', type = int, default = 42205,help='sample is 469;diginetica is 183398, short is 42205  ')
parser.add_argument('--i_N', type = int, default = 16882, help='sample is 309;diginetica is 43097,short is 16882')
parser.add_argument('--s_N', type = int, default = 42611, help='sample is 469;diginetica is 43097,short is 42611')

parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')#隐藏状态大小
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')  # [0.001, 0.0005, 0.0001] 学习率
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')#学习率衰减率
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')#学习率衰减的步数
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')#gnn传播步骤
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--embed_d', type = int, default = 100,help = 'embedding dimension')
parser.add_argument("--cuda", default = 1, type = int)

opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))#读取训练数据
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'short_diginetica':
        n_node = 16883
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset =='Tmall':
        n_node = opt.i_N + 1

    u_neigh_list_top, i_neigh_list_top ,s_neigh_list_top= het_neigh(opt)
    feature_list = get_embedding(opt)

    for i in range(len(feature_list)):
        feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()
    if opt.cuda:
        for i in range(len(feature_list)):
            feature_list[i] = feature_list[i].cuda()

    #model = trans_to_cuda(SessionGraph(opt, n_node,feature_list,u_neigh_list_top, i_neigh_list_top ,s_neigh_list_top))
    if os.path.exists(opt.data_path+"mode1l.pth"):
        model = torch.load(opt.data_path+'model.pth')
        print('加载成功！')
    else:
        print('无保存模型，将从头开始训练！')
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr= train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        torch.save(model, model.datapath)
        print('\tRecall@50:\t%.4f\tMRR@50:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

#自定义一个类，继承自Module类，并且一定要实现两个基本的函数__init__  forward
#只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现
class SessionGraph(Module):
    def __init__(self, opt, n_node,feature_list,u_neigh_list_top, i_neigh_list_top , s_neigh_list_top):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize   #100
        self.datapath = opt.data_path+"model.pth"
        self.n_node = n_node                 #310
        self.batch_size = opt.batchSize     #100
        self.nonhybrid = opt.nonhybrid
        self.embed_d = opt.embed_d
        self.feature_list =feature_list
        self.u_neigh_list_top = u_neigh_list_top
        self.i_neigh_list_top = i_neigh_list_top

        self.s_neigh_list_top = s_neigh_list_top



        self.u_content_rnn = nn.LSTM(opt.embed_d, int(opt.embed_d / 2), 1, bidirectional=True)
        self.i_content_rnn = nn.LSTM(opt.embed_d, int(opt.embed_d / 2), 1, bidirectional=True)
        self.s_content_rnn = nn.LSTM(opt.embed_d, int(opt.embed_d / 2), 1, bidirectional=True)

        self.u_neigh_rnn = nn.LSTM(opt.embed_d, int(opt.embed_d / 2), 1, bidirectional=True)
        self.i_neigh_rnn = nn.LSTM(opt.embed_d, int(opt.embed_d / 2), 1, bidirectional=True)
        self.s_neigh_rnn = nn.LSTM(opt.embed_d, int(opt.embed_d / 2), 1, bidirectional=True)

        self.u_neigh_att = nn.Parameter(torch.ones(opt.embed_d * 2, 1), requires_grad=True)
        self.i_neigh_att = nn.Parameter(torch.ones(opt.embed_d * 2, 1), requires_grad=True)
        self.s_neigh_att = nn.Parameter(torch.ones(opt.embed_d * 2, 1), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)#用于的到混合嵌入

        self.linear_final1 = nn.Linear(self.hidden_size ,2000, bias=True)
        self.linear_final2 = nn.Linear(2000, opt.i_N, bias=True)
        self.loss_function = nn.CrossEntropyLoss() #计算loss 网络输出不经 softmax 层，直接由 CrossEntropyLoss 计算交叉熵损失

        #Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重。
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)

        #将每个参数组的学习速率设置为每个step_size时间段由gamma衰减的初始lr。step_size (int) – 学习率衰减期:3  gamma (float) – 学习率衰减的乘积因子。默认值:-0.1。
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()
#********************************************************************************************
    #调整节点聚合内容，避免重复聚合数据

    def u_content_agg(self, id_batch):  # heterogeneous content aggregation
        embed_d = self.embed_d
        u_net_embed_batch = self.feature_list[0][id_batch]
        u_i_net_embed_batch = self.feature_list[1][id_batch]
        #u_s_net_embed_batch = self.feature_list[2][id_batch]
        #concate_embed = torch.cat((u_net_embed_batch, u_i_net_embed_batch,u_s_net_embed_batch), 1).view(len(id_batch[0]), 2, embed_d)
        concate_embed = torch.cat((u_i_net_embed_batch, u_i_net_embed_batch), 1).view(
            len(id_batch[0]), 2, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.u_content_rnn(concate_embed)  # 双向LSTM
        return torch.mean(all_state, 0)  # 输出input 各个元素的的均值

    def i_content_agg(self, id_batch):
        embed_d = self.embed_d
        i_net_embed_batch = self.feature_list[3][id_batch]
        #i_u_net_embed_batch = self.feature_list[4][id_batch]
        i_i_net_embed_batch = self.feature_list[5][id_batch]
        #i_s_net_embed_batch = self.feature_list[6][id_batch]
        #concate_embed = torch.cat((i_net_embed_batch, i_u_net_embed_batch, i_i_net_embed_batch,i_s_net_embed_batch), 1).view(len(id_batch[0]), 2, embed_d)
        concate_embed = torch.cat((i_net_embed_batch, i_i_net_embed_batch),
                                  1).view(len(id_batch[0]), 2, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.i_content_rnn(concate_embed)  # 双向LSTM
        return torch.mean(all_state, 0)

    def s_content_agg(self, id_batch):
        embed_d = self.embed_d
        s_net_embed_batch = self.feature_list[7][id_batch]
        s_i_net_embed_batch = self.feature_list[8][id_batch]
        concate_embed = torch.cat((s_i_net_embed_batch, s_i_net_embed_batch), 1).view(len(id_batch[0]), 2, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.s_content_rnn(concate_embed)  # 双向LSTM
        return torch.mean(all_state, 0)
    # 对同种类型的异构邻居进行训练成为一个嵌入式向量
    def node_neigh_agg(self, id_batch, node_type):  # type based neighbor aggregation with rnn
        embed_d = self.embed_d
        if node_type == 1 :
            batch_s = int(len(id_batch[0]) / 5)
        elif  node_type==2:
            batch_s = int(len(id_batch[0]) / 5)
        elif node_type == 3:
            batch_s = int(len(id_batch[0]) / 2)

        if node_type == 1:
            neigh_agg = self.u_content_agg(id_batch).view(batch_s, 5, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.u_neigh_rnn(neigh_agg)  # 聚合用户节点的不同内容
        elif node_type == 2:
            neigh_agg = self.i_content_agg(id_batch).view(batch_s, 5, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.i_neigh_rnn(neigh_agg)
        elif node_type == 3:
            neigh_agg = self.s_content_agg(id_batch).view(batch_s, 2, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.i_neigh_rnn(neigh_agg)
        neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)

        return neigh_agg

    def node_het_agg(self, id_batch, node_type):  # heterogeneous neighbor aggregation
        u_neigh_batch = [[0] * 5] * len(id_batch)
        i_neigh_batch = [[0] * 5] * len(id_batch)
        s_neigh_batch = [[0] * 2] * len(id_batch)

        for i in range(len(id_batch)):
            if node_type == 1:
                u_neigh_batch[i] = self.u_neigh_list_top[0][id_batch[i]]
                i_neigh_batch[i] = self.u_neigh_list_top[1][id_batch[i]]
                s_neigh_batch[i] = self.u_neigh_list_top[2][id_batch[i]]
            elif node_type == 2:
                u_neigh_batch[i] = self.i_neigh_list_top[0][id_batch[i]]
                i_neigh_batch[i] = self.i_neigh_list_top[1][id_batch[i]]
                s_neigh_batch[i] = self.i_neigh_list_top[2][id_batch[i]]
            elif node_type == 3:
                u_neigh_batch[i] = self.s_neigh_list_top[0][id_batch[i]]
                i_neigh_batch[i] = self.s_neigh_list_top[1][id_batch[i]]
                s_neigh_batch[i] = self.s_neigh_list_top[2][id_batch[i]]

        u_neigh_batch = np.reshape(u_neigh_batch, (1, -1))
        u_agg_batch = self.node_neigh_agg(u_neigh_batch, 1)
        i_neigh_batch = np.reshape(i_neigh_batch, (1, -1))
        i_agg_batch = self.node_neigh_agg(i_neigh_batch, 2)
        s_neigh_batch = np.reshape(s_neigh_batch, (1, -1))
        s_agg_batch = self.node_neigh_agg(s_neigh_batch, 3)

        # 注意力模型
        id_batch = np.reshape(id_batch, (1, -1))
        # 生成自身节点的嵌入向量
        if node_type == 1:
            c_agg_batch = self.u_content_agg(id_batch)
        elif node_type == 2:
            c_agg_batch = self.i_content_agg(id_batch)
        elif node_type == 3:
            c_agg_batch = self.s_content_agg(id_batch)
        # 链接操作
        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        u_agg_batch_2 = torch.cat((c_agg_batch, u_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        i_agg_batch_2 = torch.cat((c_agg_batch, i_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        s_agg_batch_2 = torch.cat((c_agg_batch, s_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

        # 计算不同嵌入的重要性
        concate_embed = torch.cat((c_agg_batch_2, u_agg_batch_2, i_agg_batch_2,s_agg_batch_2), 1).view(len(c_agg_batch), 4, self.embed_d * 2)
        if node_type == 1:
            # torch.bmm用于计算矩阵乘法
            atten_w = self.act(torch.bmm(concate_embed, self.u_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.u_neigh_att.size())))
        elif node_type == 2:
            atten_w = self.act(torch.bmm(concate_embed, self.i_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.i_neigh_att.size())))
        elif node_type == 3:
            atten_w = self.act(torch.bmm(concate_embed, self.s_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.s_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)

        # weighted combination
        concate_embed = torch.cat((c_agg_batch, u_agg_batch, i_agg_batch,s_agg_batch), 1).view(len(c_agg_batch), 4, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)
        return weight_agg_batch

    #将inputs中的每个会话的物品，进行异构得到向量表示100*7->100 *7 *100
    def het_agg(self, inputs):
        node = []
        for s in inputs:
            for i in s:
                item = i.data.item()
                if item != 0 and item not in node:
                    node.append(item)
        node = np.array(node)
        i_agg_embed = self.node_het_agg(node, 2)

        nodes = {}
        for i in range(len(node)):
            nodes[node[i]] = i_agg_embed[i]
        outputs = torch.zeros([1, 100]).cuda()
        flag = 0
        zeros = torch.zeros([1, self.embed_d]).cuda()
        for s in inputs:
            for i in s:
                item = i.data.item()
                if item == 0:
                    if flag == 0:
                        flag = 1
                        continue
                    outputs = torch.cat((outputs, zeros), 0)
                else:
                    outputs = torch.cat((outputs, nodes[item].view(1, 100)), 0)
        outputs = outputs.view(len(inputs), len(inputs[0]), 100).float()
        return outputs


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    #用于测试计算得分
    def compute_scores(self, hidden, mask):
        #考虑全局和局部偏好
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:#得到混合嵌入
            a = self.linear_transform(torch.cat([a, ht], 1))
        #b = self.embedding.weight[1:]
        #scores = torch.matmul(a, b.transpose(1, 0))
        scores=self.linear_final1(a)
        scores = self.linear_final2(scores)
        return scores

    def forward(self, inputs):
        hidden = self.het_agg(inputs)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())#把张量转化为variable
    items = trans_to_cuda(torch.Tensor(items).long())
    # A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    hidden = model(items)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
      #对学习率进行调整
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)#把训练数据1205个会话分13批进行训练，每批100
    for i, j in zip(slices, np.arange(len(slices))):#每次100个会话
        # break
        model.optimizer.zero_grad()#把模型参数梯度设为0
        targets, scores = forward(model, i, train_data)#前向传播 scores得分矩阵
        targets = trans_to_cuda(torch.Tensor(targets).long())#构造variable
        loss = model.loss_function(scores, targets - 1)#损失函数
        loss.backward()#反向传播
        model.optimizer.step()#模型更新
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)

    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(50)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100


    return hit, mrr

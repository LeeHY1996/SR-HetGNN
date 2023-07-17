import random
import re
from collections import Counter
from itertools import islice
import networkx as nx
import numpy as np




def data_masks(all_usr_pois, item_tail):
	us_lens = [len(upois) for upois in all_usr_pois]
	len_max = max(us_lens)
	us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
	us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
	return us_pois, us_msks, len_max

#对于数据整体内容进行操作 通过下标操作 np.arange获得对应的索引采用的列表 进行对应的切分操作
def split_validation(train_set, valid_portion):
	train_set_x, train_set_y = train_set
	n_samples = len(train_set_x)
	sidx = np.arange(n_samples, dtype='int32')
	np.random.shuffle(sidx)
	n_train = int(np.round(n_samples * (1. - valid_portion)))
	valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
	valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
	train_set_x = [train_set_x[s] for s in sidx[:n_train]]
	train_set_y = [train_set_y[s] for s in sidx[:n_train]]

	return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

#定义数据类
class Data():
	def __init__(self, data, shuffle=False, graph=None):
		inputs = data[0]                                    # 会话序列
		inputs, mask, len_max = data_masks(inputs, [0])
		self.inputs = np.asarray(inputs)
		self.mask = np.asarray(mask) #[1 1 0 ... 0 0 0] [1 0 0 ... 0 0 0]...
		self.len_max = len_max
		self.targets = np.asarray(data[1])                  #预测序列
		self.length = len(inputs)
		self.shuffle = shuffle


	def generate_batch(self, batch_size):
		if self.shuffle:#训练集为true
			shuffled_arg = np.arange(self.length)
			np.random.shuffle(shuffled_arg)     #s随机生成列表
			#每一次训练会话的组合都是随机的
			self.inputs = self.inputs[shuffled_arg] #随机矩阵
			self.mask = self.mask[shuffled_arg]
			self.targets = self.targets[shuffled_arg]

		n_batch = int(self.length / batch_size)
		if self.length % batch_size != 0:
			n_batch += 1
		slices = np.split(np.arange(n_batch * batch_size), n_batch)
		slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
		return slices

	def get_slice(self, i):
		inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]#100个会话
		items, n_node, alias_inputs = [], [], []
		for u_input in inputs:
			n_node.append(len(np.unique(u_input)))
		max_n_node = np.max(n_node)#得到max_n_node 100会话中item数量最大值 从而确定邻接矩阵大小
		for u_input in inputs:
			node = np.unique(u_input)#去除重复数字并排序 ，获取item号
			items.append(node.tolist() + (max_n_node - len(node)) * [0])
			# u_A = np.zeros((max_n_node, max_n_node))
			for i in np.arange(len(u_input) - 1):
				if u_input[i + 1] == 0:
					break
			# 	u = np.where(node == u_input[i])[0][0]
			# 	v = np.where(node == u_input[i + 1])[0][0]
			# 	u_A[u][v] = 1
			# u_sum_in = np.sum(u_A, 0)
			# u_sum_in[np.where(u_sum_in == 0)] = 1
			# u_A_in = np.divide(u_A, u_sum_in)
			# u_sum_out = np.sum(u_A, 1)
			# u_sum_out[np.where(u_sum_out == 0)] = 1
			# u_A_out = np.divide(u_A.transpose(), u_sum_out)
			# u_A = np.concatenate([u_A_in, u_A_out]).transpose()
			# A.append(u_A) #会话图链接矩阵，max_node,max_node*2 文章中有提到
			alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
		return alias_inputs,  items, mask, targets



#得到每个节点的异构邻居
def het_neigh(opt):
	u_N = opt.u_N + 1 #sample = 470 diginetica is 183399
	i_N = opt.i_N + 1 #sample = 309;diginetica = 43098
	s_N = opt.s_N + 1
	u_neigh_list = [[[] for i in range(u_N)] for j in range(3)]
	i_neigh_list = [[[] for i in range(i_N)] for j in range(3)]
	s_neigh_list = [[[] for i in range(s_N)] for j in range(3)]

	het_neigh_train_f = open(opt.data_path + "het_restart_neigh.txt", "r")
	for line in het_neigh_train_f:
		line = line.strip()
		node_id = re.split(':', line)[0]

		neigh = re.split(':', line)[1]
		neigh_list = re.split(',', neigh)
		if node_id[0] == 'u' and len(node_id) > 1:
			for j in range(len(neigh_list)):
				if neigh_list[j][0] == 'u':
					u_neigh_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif neigh_list[j][0] == 'i':
					u_neigh_list[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif neigh_list[j][0] == 's':
					u_neigh_list[2][int(node_id[1:])].append(int(neigh_list[j][1:]))

		elif node_id[0] == 'i' and len(node_id) > 1:
			for j in range(len(neigh_list)):
				if neigh_list[j][0] == 'u':
					i_neigh_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif neigh_list[j][0] == 'i':
					i_neigh_list[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif neigh_list[j][0] == 's':
					i_neigh_list[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
		elif node_id[0] == 's' and len(node_id) > 1:
			for j in range(len(neigh_list)):
				if neigh_list[j][0] == 'u':
					s_neigh_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif neigh_list[j][0] == 'i':
				    s_neigh_list[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif neigh_list[j][0] == 's':
					s_neigh_list[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
	het_neigh_train_f.close()


	u_neigh_list_top = [[[] for i in range(u_N)] for j in range(3)]
	i_neigh_list_top = [[[] for i in range(i_N)] for j in range(3)]
	s_neigh_list_top = [[[] for i in range(s_N)] for j in range(3)]
	top_k = [5, 5 ,2]
	for i in range(u_N):
		for j in range(3):
			u_neigh_list_temp = Counter(u_neigh_list[j][i])  # Counter 类用于找出一个序列中出现次数最多的元素
			top_list = u_neigh_list_temp.most_common(top_k[j])  # 选取top n的功能
			neigh_size = top_k[j]

			for k in range(len(top_list)):
				u_neigh_list_top[j][i].append(int(top_list[k][0]))
			if len(u_neigh_list_top[j][i]) and len(u_neigh_list_top[j][i]) < neigh_size:
				for l in range(len(u_neigh_list_top[j][i]), neigh_size):
					u_neigh_list_top[j][i].append(random.choice(u_neigh_list_top[j][i]))
	for i in range(i_N):
		for j in range(3):
			i_neigh_list_train_temp = Counter(i_neigh_list[j][i])
			top_list = i_neigh_list_train_temp.most_common(top_k[j])
			neigh_size = top_k[j]
			for k in range(len(top_list)):
				i_neigh_list_top[j][i].append(int(top_list[k][0]))
			if len(i_neigh_list_top[j][i]) and len(i_neigh_list_top[j][i]) < neigh_size:
				for l in range(len(i_neigh_list_top[j][i]), neigh_size):
					i_neigh_list_top[j][i].append(random.choice(i_neigh_list_top[j][i]))
	for i in range(s_N):
		for j in range(3):
			s_neigh_list_train_temp = Counter(s_neigh_list[j][i])
			top_list = s_neigh_list_train_temp.most_common(top_k[j])
			neigh_size = top_k[j]
			for k in range(len(top_list)):
				s_neigh_list_top[j][i].append(int(top_list[k][0]))
			if len(s_neigh_list_top[j][i]) and len(s_neigh_list_top[j][i]) < neigh_size:
				for l in range(len(s_neigh_list_top[j][i]), neigh_size):
					s_neigh_list_top[j][i].append(random.choice(s_neigh_list_top[j][i]))

	return u_neigh_list_top,i_neigh_list_top,s_neigh_list_top

def get_embedding(opt):
	u_N = opt.u_N + 1
	i_N = opt.i_N + 1
	s_N = opt.s_N + 1
	dim = 100
	u_net_embed = np.zeros((u_N, dim))
	i_net_embed = np.zeros((i_N, dim))
	s_net_embed = np.zeros((s_N, dim))

	net_e_f = open(opt.data_path + "node_net_embedding.txt", "r")
	for line in islice(net_e_f, 1, None):
		line = line.strip()
		index = re.split(' ', line)[0]
		if len(index) and (index[0] == 'u'  or index[0] == 'i' or index[0] == 's'):
			embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
			if index[0] == 'u':
				u_net_embed[int(index[1:])] = embeds
			elif index[0] == 'i':
				i_net_embed[int(index[1:])] = embeds
			elif index[0] == 's':
				s_net_embed[int(index[1:])] = embeds

	net_e_f.close()

	i_u_net_embed = np.zeros((i_N, dim))
	i_u_f = open(opt.data_path  + "i_u.txt", "r")
	for line in i_u_f:
		line = line.strip().split(':')
		i_id = int(line[0])
		u_ids = line[1].split(',')
		for i in u_ids:
			u_id = int(i)
			i_u_net_embed[i_id] = np.add(i_u_net_embed[i_id],u_net_embed[u_id])
		i_u_net_embed[i_id] = i_u_net_embed[i_id]/len(u_ids)
	i_u_f.close()

	i_i_net_embed = np.zeros((i_N, dim))
	i_i_f = open(opt.data_path  + "i_i.txt", "r")
	for line in i_i_f:
		line = line.strip().split(':')
		i_id = int(line[0])
		ii_ids = line[1].split(',')
		for i in ii_ids:
			ii_id = int(i)
			i_i_net_embed[i_id] = np.add(i_i_net_embed[i_id], i_net_embed[ii_id])
		i_i_net_embed[i_id] = i_i_net_embed[i_id] / len(ii_ids)
	i_i_f.close()

	i_s_net_embed = np.zeros((i_N, dim))
	i_s_f = open(opt.data_path + "i_s.txt", "r")
	for line in i_s_f:
		line = line.strip().split(':')
		i_id = int(line[0])
		s_ids = line[1].split(',')
		for i in s_ids:
			s_id = int(i)
			i_s_net_embed[i_id] = np.add(i_s_net_embed[i_id], s_net_embed[s_id])
		i_s_net_embed[i_id] = i_s_net_embed[i_id] / len(s_ids)
	i_s_f.close()

	u_i_net_embed = np.zeros((u_N, dim))
	u_i_f = open(opt.data_path  + "u_i.txt", "r")
	for line in u_i_f:
		line = line.strip().split(':')
		u_id = int(line[0])
		i_ids = line[1].split(',')
		for i in i_ids:
			i_id = int(i)
			u_i_net_embed[u_id] = np.add(u_i_net_embed[u_id], i_net_embed[i_id])
		u_i_net_embed[u_id] = u_i_net_embed[u_id] / len(i_ids)
	u_i_f.close()

	u_s_net_embed = np.zeros((u_N, dim))
	u_s_f = open(opt.data_path + "u_s.txt", "r")
	for line in u_s_f:
		line = line.strip().split(':')
		u_id = int(line[0])
		s_ids = line[1].split(',')
		for i in s_ids:
			s_id = int(i)
			u_s_net_embed[u_id] = np.add(u_s_net_embed[u_id], s_net_embed[s_id])
		u_s_net_embed[u_id] = u_s_net_embed[u_id] / len(s_ids)
	u_s_f.close()

	s_i_net_embed = np.zeros((s_N, dim))
	s_i_f = open(opt.data_path + "s_i.txt", "r")
	for line in s_i_f:
		line = line.strip().split(':')
		s_id = int(line[0])
		i_ids = line[1].split(',')
		for i in i_ids:
			i_id = int(i)
			s_i_net_embed[s_id] = np.add(s_i_net_embed[s_id], i_net_embed[i_id])
		s_i_net_embed[s_id] = s_i_net_embed[s_id] / len(i_ids)
	s_i_f.close()




	# return [u_net_embed ,u_i_net_embed , u_s_net_embed,i_net_embed,i_u_net_embed , i_i_net_embed , i_s_net_embed ,s_net_embed ,s_i_net_embed]
	return [u_net_embed ,u_i_net_embed , [],i_net_embed,[] , i_i_net_embed , [] ,s_net_embed ,s_i_net_embed]
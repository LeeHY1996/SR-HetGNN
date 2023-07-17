import argparse
import random
from collections import Counter
import re
from gensim.models import Word2Vec
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = 'Tmall/',
				   help='path to data')
parser.add_argument('--u_N', type = int, default = 13543,
			   help='sample is 469;diginetica is 183398;short is 51691 ')
parser.add_argument('--i_N', type = int, default = 3697,
			   help='sample is 309;diginetica is 43097 ;short is 17339')
parser.add_argument('--s_N', type = int, default = 21578,
			   help='sample is 469;diginetica is 43097 ;short is 53928')
parser.add_argument('--walk_n', type = int, default = 10,
			   help='number of walk per root node')
parser.add_argument('--walk_L', type = int, default = 100,
			   help='length of each walk')
parser.add_argument('--u_i', type = str, default = 'u_i.txt',
			   help='u_i.txt')
parser.add_argument('--u_s', type = str, default = 'u_s.txt',
			   help='u_s.txt')
parser.add_argument('--i_u', type = str, default = 'i_u.txt',
			   help='i_u.txt')
parser.add_argument('--i_i', type = str, default = 'i_i.txt',
			   help='i_i.txt')
parser.add_argument('--i_s', type = str, default = 'i_s.txt',
			   help='i_s.txt')
parser.add_argument('--s_i', type = str, default = 's_i.txt',
			   help='s_i.txt')

parser.add_argument('--dim', type = int, default = 100,
			   help='dim')
opt = parser.parse_args()
print(opt)

def getdata():
	u_i = {}
	u_if = open(opt.data_path+opt.u_i,'r')
	for i in u_if:
		d=i.strip('\n').split(':')
		u_i[int(d[0])] = []
		for h in d[1].split(','):
			u_i[int(d[0])].append('i'+h)
	u_if.close()

	u_s = {}
	u_sf = open(opt.data_path + opt.u_s, 'r')
	for i in u_sf:
		d = i.strip('\n').split(':')
		u_s[int(d[0])] = []
		for h in d[1].split(','):
			u_s[int(d[0])].append('s' + h)
	u_sf.close()

	i_u = {}
	i_uf = open(opt.data_path+opt.i_u, 'r')
	for i in i_uf:
		d = i.strip('\n').split(':')
		i_u[int(d[0])] = []
		for h in d[1].split(','):
			i_u[int(d[0])].append('u'+h)

	i_uf.close()

	i_i = {}
	i_if = open(opt.data_path+opt.i_i, 'r')
	for i in i_if:
		d = i.strip('\n').split(':')
		i_i[int(d[0])] = []
		for h in d[1].split(','):
			i_i[int(d[0])].append('i' + h)
	i_if.close()

	i_s = {}
	i_sf = open(opt.data_path + opt.i_s, 'r')
	for i in i_sf:
		d = i.strip('\n').split(':')
		i_s[int(d[0])] = []
		for h in d[1].split(','):
			i_s[int(d[0])].append('s' + h)
	i_sf.close()

	s_i = {}
	s_if = open(opt.data_path + opt.s_i, 'r')
	for i in s_if:
		d = i.strip('\n').split(':')
		s_i[int(d[0])] = []
		for h in d[1].split(','):
			s_i[int(d[0])].append('i' + h)
	s_if.close()

	i_neigh_list = [[] for k in range(len(i_u)+1)]
	for i in i_u.keys():

		if i in i_u.keys():
			i_neigh_list[i] += i_u[i]
		if i in i_i.keys():
			i_neigh_list[i] += i_i[i]
		if i in i_s.keys():
			i_neigh_list[i] += i_s[i]
		#print(i,i_neigh_list[i])


	u_neigh_list = [[] for k in range(len(u_i)+1)]
	for i in u_i.keys():
		if i in u_i.keys():
			u_neigh_list[i] += u_i[i]
		if i in u_s.keys():
			u_neigh_list[i] += u_s[i]
		# print(i,u_neigh_list[i])

	s_i_list = [[] for k in range(len(s_i) + 1)]
	for i in s_i.keys():
		if i in s_i.keys():
			# print(u_i[i])
			s_i_list[i] += s_i[i]
		#print(i, s_i_list[i])

	return u_neigh_list,i_neigh_list,s_i_list

#进行简单随机游走
def gen_het_rand_walk(u_neigh_list, i_neigh_list,s_i_list):
	het_walk_f = open(opt.data_path + "het_random_walk.txt", "w")
	#print len(self.p_neigh_list_train)
	for i in range(opt.walk_n):#进行10次随机游走
		for j in range(len(u_neigh_list)):
			if len(u_neigh_list[j]):
				curNode = "u" + str(j)#随机游走的起点
				het_walk_f.write(curNode + " ")
				for l in range(opt.walk_L - 1):#随机游走长度30
					if curNode[0] == "u":
						curNode = int(curNode[1:])
						curNode = random.choice(u_neigh_list[curNode])
						het_walk_f.write(curNode + " ")
					elif curNode[0] == "i":
						curNode = int(curNode[1:])
						curNode = random.choice(i_neigh_list[curNode])
						het_walk_f.write(curNode + " ")
					elif curNode[0] == "s":
						curNode = int(curNode[1:])
						curNode = random.choice(s_i_list[curNode])
						het_walk_f.write(curNode + " ")
				het_walk_f.write("\n")
	het_walk_f.close()

#进行基于重启的随机游走
#在随机游走的过程中可能会回到起点，同时控制每种类型节点的数量，确保每一种节点都能够游走到
def het_walk_restart(u_neigh, i_neigh, s_i):
	u_N = opt.u_N + 1
	i_N = opt.i_N + 1
	s_N = opt.s_N + 1
	u_neigh_list = [[] for k in range(u_N)]
	i_neigh_list = [[] for k in range(i_N)]
	s_neigh_list = [[] for k in range(s_N)]
	#generate neighbor set via random walk with restart
	node_n = [u_N, i_N , s_N]
	for i in range(3):
		for j in range(node_n[i]):
			if i == 0:
				neigh_temp = u_neigh[j]
				neigh_train = u_neigh_list[j]
				curNode = "u" + str(j)
			elif i == 1 :
				neigh_temp = i_neigh[j]
				neigh_train = i_neigh_list[j]
				curNode = "i" + str(j)
			else:
				neigh_temp = s_i[j]
				neigh_train = s_neigh_list[j]
				curNode = "s" + str(j)

			if len(neigh_temp):
				neigh_L = 0
				u_L = 0
				i_L = 0
				s_L = 0
				while neigh_L < 500: #maximum neighbor size = 100

					rand_p = random.random() #return p
					if rand_p > 0.7:
						if curNode[0] == "u":
							curNode = random.choice(u_neigh[int(curNode[1:])])
							if curNode[0] == 'i' and i_L < 300: #size constraint (make sure each type of neighobr is sampled)
								neigh_train.append(curNode)
								neigh_L += 1
								i_L += 1
							elif curNode[0] == 's' and s_L < 100:
								neigh_train.append(curNode)
								neigh_L += 1
								s_L += 1

						elif curNode[0] == "i":
							curNode = random.choice(i_neigh[int(curNode[1:])])
							if curNode[0] == 'u' and u_L < 100  :
							# if curNode[0] == 'u' and u_L < 40:
								neigh_train.append(curNode)
								neigh_L += 1
								u_L += 1
							elif curNode[0] == 'i':
								if i_L < 300:
									neigh_train.append(curNode)
									neigh_L += 1
									i_L += 1
							elif curNode[0] == 's':
								if s_L < 100:
									neigh_train.append(curNode)
									neigh_L += 1
									s_L += 1
						elif curNode[0] == "s":
							curNode = random.choice(s_i[int(curNode[1:])])
							if i_L < 300:
								neigh_train.append(curNode)
								neigh_L += 1
								i_L += 1
					else:
						if i == 0:
							curNode = ('u' + str(j))
						elif i==1:
							curNode = ('i' + str(j))
						else:
							curNode = ('s' + str(j))
	for i in range(3):
		for j in range(node_n[i]):
			if i == 0:
				u_neigh_list[j] = list(u_neigh_list[j])
			elif i == 1:
				i_neigh_list[j] = list(i_neigh_list[j])
			else :
				s_neigh_list[j] = list(s_neigh_list[j])

	neigh_f = open(opt.data_path + "het_restart_neigh.txt", "w")
	for i in range(3):
		for j in range(node_n[i]):
			if i == 0:
				neigh_train = u_neigh_list[j]
				curNode = "u" + str(j)
			elif i==1:
				neigh_train = i_neigh_list[j]
				curNode = "i" + str(j)
			else:
				neigh_train = s_neigh_list[j]
				curNode = "s" + str(j)

			if len(neigh_train):
				neigh_f.write(curNode + ":")
				for k in range(len(neigh_train) - 1):
					neigh_f.write(neigh_train[k] + ",")
				neigh_f.write(neigh_train[-1] + "\n")
	neigh_f.close()



#生成预训练节点，使用word2vec模型，将每一个节点当做一个词，生成词向量
def generat_node_walk_embedding():
	dimen = opt.dim
	window = 5
	walks=[]
	#inputfile = open("../data/academic_test/meta_random_walk_APVPA_test.txt","r")
	inputfile = open(opt.data_path+"/het_random_walk.txt", "r")
	for line in inputfile:
		path = []
		node_list=re.split(' ',line.strip('\n').strip(' '))
		# print(node_list)
		for i in range(len(node_list)):
			path.append(node_list[i])
		walks.append(path)
	inputfile.close()
	model = Word2Vec(walks, size=dimen, window=window, min_count=0, workers=2, sg=1, hs=0, negative=5)
	model.wv.save_word2vec_format(opt.data_path+"/node_net_embedding.txt")  # 生成预训练节点嵌入 100维


print('读取数据')
u_neigh_list,i_neigh_list,s_i_list=getdata()
#
print('进行简单随机游走')
gen_het_rand_walk(u_neigh_list,i_neigh_list,s_i_list)

print('进行基于重启的随机游走')
het_walk_restart(u_neigh_list,i_neigh_list,s_i_list)

# u_neigh_list_top,i_neigh_list_top=het_neigh()
# for i in range(0,310):
# 	print(i,'u',i_neigh_list_top[0][i])
# 	print(i,'i', i_neigh_list_top[1][i])
# 308 u [463, 464, 159]
# 308 i [65, 308, 171]
# 309 u [465, 465, 465]
# 309 i [309, 309, 309]
print('生成预嵌入向量')
generat_node_walk_embedding()
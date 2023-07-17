#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)
dataset = 'data/Tmall_buys.csv'

if opt.dataset == 'diginetica':
    dataset = 'data/train-item-views.csv'
elif opt.dataset == 'Tmall':
    dataset = 'data/Tmall_buys.csv'
elif opt.dataset == 'short_diginetica':
    dataset = 'data/short-item-views.csv'


#*****************************************************************
print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=',')
    sess_clicks = {}
    sess_date = {}
    sess_userid={}
    ctr = 0
    curid = -1
    curuserid=-1
    curdate = None
    for data in reader:
        sessid = data['sessionId'] #会话号
        userid = data['user_id']  # 会话号
        if curdate and not curid == sessid:

            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = int(data['time_stamp'])
            sess_date[curid] = date
            sess_userid[curid] = curuserid
        curid = sessid
        curuserid = userid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['item_id']
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = int(data['time_stamp'])

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1

    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        curdate = int(data['time_stamp'])

    sess_date[curid] = date
    sess_userid[curid] = userid
print("-- Reading data @ %ss" % datetime.datetime.now())
# print(sess_date) #{'1': 1462723200.0, '2': 1462723200.0, '4': 1462723200.0, '5': 1462723200.0, '6': 1462723200.0, '7': 1462723200.0, '8': 1460217600.0,}每个会话开始时间
# print(sess_userid)
#print(sess_clicks) {'1': ['9654', '33043', '32118', '12352', '35077', '36118', '81766', '129055', '31331', '32627'],
#                    '2': ['100747', '35606', '32971', '36246', '32754', '10657', '36246', '35606', '3147', '196110'],}
#                    每个会话的物品

# 删除长度为1的会话
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
        del sess_userid[s]

# 计算每个项目出现的次数
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

#根据出现的次数，对项目item进行排序
sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))
#print(sorted_counts)  ('32118', 1), ('81766', 1), ('129055', 1), ('31331', 1)
length = len(sess_clicks) #item数量

for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
        del sess_userid[s]
    else:
        sess_clicks[s] = filseq
#到此处理的数据，无长度为1的会话， 出现次数小于
# 5的物品
#print(len(sess_clicks),len(sess_date)) 会话长度525


# 根据日期划分测试集
dates = list(sess_date.items())
#print(dates)  [('2', 1462723200.0), ('5', 1462723200.0), ('7', 1462723200.0), ('12', 1459785600.0)]会话
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 分割测试数据和训练数据
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 5  #7天

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)#
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# 按日期对会话排序
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print('训练集会话长度',len(tra_sess))    # 186670    # 7966257
print('测试集会话长度',len(tes_sess))    # 15979     # 15324
# print(tra_sess)
# print(tes_sess)
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())



# 将训练的会话转换为序列并对item重新编号从1开始
item_dict = {} #对item重新编号
userid_dict = {}
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    train_userids = []
    item_ctr = 1
    userid_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        userid = sess_userid[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        if userid in userid_dict:
            userid = userid_dict[userid]
        else:
            if userid == 'NA':
                userid = userid_ctr
            else:
                userid_dict[userid] = userid_ctr
                userid = userid_ctr
            userid_ctr += 1

        train_ids += [s]
        train_userids += [userid]
        train_dates += [date]
        train_seqs += [outseq]
    print('训练集：',item_ctr)
    return train_ids, train_dates, train_seqs,train_userids

# 将测试会话转换为序列，而忽略训练集中未出现的项目
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    test_userids =[]
    # test_userid=[]
    userid_ctr = len(userid_dict) + 1
    for s, date in tes_sess:
        seq = sess_clicks[s]
        userid = sess_userid[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq)< 2 or (userid not in userid_dict and opt.dataset!='sample') :
            continue
        if userid in userid_dict:
            userid = userid_dict[userid]
        else:
            if userid == 'NA':
                userid = userid_ctr
            else:
                userid_dict[userid] = userid_ctr
                userid = userid_ctr
            userid_ctr += 1
        test_ids += [s]
        test_userids += [userid]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs,test_userids

#会话编号 ， 时间 ，  物品序列,用户id
tra_ids, tra_dates, tra_seqs ,tra_userids = obtian_tra()
tes_ids, tes_dates, tes_seqs ,tes_userids= obtian_tes()
print(len(tes_ids))
print(tes_seqs)
# print(tes_userids)
#['1864', '1867', '1868', '1879',
#[1464192000.0, 1464192000.0, 1464192000.0, 1464192000.0,
#[[282, 282], [281, 308, 281], [58, 58, 58, 230, 230, 230, 246, 230, 230], [200, 200],
if not os.path.exists(opt.dataset):
    os.makedirs(opt.dataset)
f = open(opt.dataset+'/u_i.txt','w')
u_i={}
for i in range(0,len(tra_userids)):
    seq = tra_seqs[i]
    if tra_userids[i] in u_i.keys():
        for h in seq:
            if h not in u_i[tra_userids[i]]:
                u_i[tra_userids[i]].append(h)
    else:
        u_i[tra_userids[i]]=[]
        for h in seq:
            if h not in u_i[tra_userids[i]]:
                u_i[tra_userids[i]].append(h)
# del u_i['NA']
for i in u_i.keys():
    s = ''
    s = str(i)+':'
    for h in u_i[i]:
        s=s + str(h)
        if h !=u_i[i][-1]:
            s=s+','
    s = s + '\n'
    #print(s)
    f.write(s)
f.close()
f = open(opt.dataset+'/i_u.txt','w')
i_u={}
for i in range(0,len(tra_seqs)):
    seq = tra_seqs[i]
    for h in seq:
        if h in i_u.keys():
            if tra_userids[i] not in i_u[h] :
                i_u[h].append(tra_userids[i])
        else:
            i_u[h]=[]
            # if tra_userids[i]!='NA':
            i_u[h].append(tra_userids[i])

for i in i_u.keys():
    if len(i_u[i])!=0:
        s = ''
        s = str(i)+':'
        for h in i_u[i]:
            s=s + str(h)
            if h !=i_u[i][-1]:
                s=s+','
        s = s + '\n'
        #print(s)
        f.write(s)
f.close()
f = open(opt.dataset+'/i_i_n.txt','w')
i_i={}
for i in range(0,len(tra_seqs)):
    seq = tra_seqs[i]
    for h in range(0,len(seq)-1):
        if seq[h] in i_i.keys():
            i_i[seq[h]].append(seq[h+1])
        else:
            i_i[seq[h]]=[]
            i_i[seq[h]].append(seq[h+1])

for i in i_i.keys():
    s = ''
    s = str(i)+':'
    num=0
    for h in range(0,len(i_i[i])-1):
        s=s + str(i_i[i][num])+','
        num=num+1
    s = s + str(i_i[i][num])
    s = s + '\n'
   # print(s)
    f.write(s)
f.close()
f = open(opt.dataset+'/i_i.txt','w')
i_i={}
for i in range(0,len(tra_seqs)):
    seq = tra_seqs[i]
    for h in range(0,len(seq)-1):
        if seq[h] in i_i.keys():
            if seq[h+1] not in i_i[seq[h]]:
                i_i[seq[h]].append(seq[h+1])
        else:
            i_i[seq[h]]=[]
            i_i[seq[h]].append(seq[h+1])
for i in i_i.keys():
    s = ''
    s = str(i)+':'
    num=0
    for h in range(0,len(i_i[i])-1):
        s=s + str(i_i[i][num])+','
        num=num+1
    s = s + str(i_i[i][num])
    s = s + '\n'
    f.write(s)
f.close()

f = open(opt.dataset+'/i_s.txt','w')
i_s={}
for i in range(0,len(tra_seqs)):
    seq = tra_seqs[i]
    for h in seq:
        if h in i_s.keys():
            if i+1 not in i_s[h]:
                i_s[h].append(i+1)
        else:
            i_s[h]=[]
            i_s[h].append(i+1)

for i in i_s.keys():
    if len(i_s[i])!=0:
        s = ''
        s = str(i)+':'
        num = 0
        for h in range(0, len(i_s[i]) - 1):
            s = s + str(i_s[i][num]) + ','
            num = num + 1
        s = s + str(i_s[i][num])
        s = s + '\n'
        f.write(s)
f.close()

f = open(opt.dataset+'/u_s.txt','w')
u_s={}
for i in range(0,len(tra_userids)):
    h = tra_userids[i]
    if h in u_s.keys():
        if i+1 not in u_s[h]:
            u_s[h].append(i+1)
    else:
        u_s[h]=[]
        u_s[h].append(i+1)

for i in u_s.keys():
    if len(u_s[i])!=0:
        s = ''
        s = str(i)+':'
        num = 0
        for h in range(0, len(u_s[i]) - 1):
            s = s + str(u_s[i][num]) + ','
            num = num + 1
        s = s + str(u_s[i][num])
        s = s + '\n'
        f.write(s)
f.close()

f = open(opt.dataset+'/s_i.txt','w')
s_i={}
for i in range(0,len(tra_seqs)):
    seq = tra_seqs[i]
    s_i[i]=[]
    for h in seq:
        if h not in s_i[i]:
            s_i[i].append(h)


for i in s_i.keys():
    if len(s_i[i])!=0:
        s = ''
        s = str(i+1)+':'
        for h in s_i[i]:
            s=s + str(h)
            if h !=s_i[i][-1]:
                s=s+','
        s = s + '\n'
        f.write(s)
f.close()



def process_seqs(iseqs, idates,userids):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    users = []

    for id, seq, date ,userid in zip(range(len(iseqs)), iseqs, idates,userids):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
            users += [userid]
    return out_seqs, out_dates, labs, ids,users


tr_seqs, tr_dates, tr_labs, tr_ids ,tr_users = process_seqs(tra_seqs, tra_dates,tra_userids)
te_seqs, te_dates, te_labs, te_ids ,te_users= process_seqs(tes_seqs, tes_dates,tes_userids)
tra = (tr_seqs, tr_labs,tr_users)
tes = (te_seqs, te_labs,te_users)
print(len(te_seqs))
#print(te_labs)

print('训练集会话：',len(tr_seqs))
print('测试集会话：',len(te_seqs))

# f=open('tr.csv','w')
# f.write('uid'+','+ 'iid'+','+'rating'+','+'timestamp')
# f.write('\n')
# u={}
# for i in range(len(tr_seqs)):
#     for s in tr_seqs[i]:
#         key = str(tr_users[i])+','+str(s)
#         if key in u.keys():
#             u[key][0]=u[key][0]+1
#         else:
#             u[key]=[1,tr_dates[i]]
# for key in u.keys():
#     [user,item]=key.split(',')
#     f.write(user + ',' + item + ',' + str(u[key][0])+',' + str(u[key][1])+'\n')
# f.close()



#[1,2,3]
# print('序列：',tr_seqs)
# #[1,2],[1]
# print('预测：',tr_labs)
#[3],[2]

all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'short_diginetica':
    if not os.path.exists('short_diginetica'):
        os.makedirs('short_diginetica')
    pickle.dump(tra, open('short_diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('short_diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('short_diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('Tmall'):
        os.makedirs('Tmall')
    pickle.dump(tra, open('Tmall/train.txt', 'wb'))
    pickle.dump(tes, open('Tmall/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('Tmall/all_train_seq.txt', 'wb'))

print('Done.')

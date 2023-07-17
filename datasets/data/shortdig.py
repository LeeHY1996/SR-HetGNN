import csv


f = open('train-item-views.csv','r')
datas = []
data = f.readline()
data = data.strip('\n').split(';')
datas.append(data)
for data in f:
    data = data.strip('\n').split(';')  # 会话号
    if data[1]!='NA':
        datas.append(data)
f.close()


f = open('short-item-views.csv','w',newline='')
writer = csv.writer(f,delimiter=';')
writer.writerows(datas)
f.close()
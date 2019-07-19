import matplotlib.pyplot as plt
import numpy as np
import csv

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

csv_file = open('p.csv',encoding='utf-8')  # 打开文件
csv_reader_lines = csv.reader(csv_file)  # 用csv.reader读文件
dataset = []
for one_line in csv_reader_lines:
    dataset.append(one_line)  # 逐行将读到的文件存入python的列表
dataset = np.array(dataset)  # 将python列表转化为ndarray
    # print(dataset,type(dataset))
dataset = dataset.astype('float64')
# print(dataset)
Xa = dataset[:13]
Xb = dataset[13:26]
Xc = dataset[26:39]
Xd = dataset[39:]

# plt.figure()
# plt.xlim(-25,25)
# plt.ylim(-30,30)
# plt.scatter(dataset[:13,0],dataset[:13,1],label = "A",color = (0,0,0))
# plt.scatter(dataset[13:26,0],dataset[13:26,1],label="B",color = (0,1,0))
# plt.scatter(dataset[26:39,0],dataset[26:39,1],label = "C",color = (0,0,1))
# plt.scatter(dataset[39:,0],dataset[39:,1],label = "D", color=(1,0,0))
# plt.xlabel("X/公里")
# plt.ylabel("Y/公里")
# plt.title("某市人口分部图")
# plt.legend(loc = "best")
# plt.savefig('map.jpg')
# plt.show()

import numpy as np
import csv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class NaiveBayes:
    def __init__(self, lamb=1):
        self.lamb = lamb  # 贝叶斯估计的参数
        self.prior = dict()  # 存储先验概率
        self.conditional = dict()  # 存储条件概率

    def training(self, features, target):
        """
        根据朴素贝叶斯算法原理,使用 贝叶斯估计 计算先验概率和条件概率
        特征集集为离散型数据,预测类别为多元.  数据集格式为np.array
        :param features: 特征集m*n,m为样本数,n为特征数
        :param target: 标签集m*1
        :return: 不返回任何值,更新成员变量
        """
        features = np.array(features)
        target = np.array(target).reshape(features.shape[0], 1)
        # print(target)
        m, n = features.shape
        # print(m,n)
        labels = Counter(target.flatten().tolist())  # 计算各类别的样本个数
        # print(labels)
        k = len(labels.keys())  # 类别数
        for label, amount in labels.items():
            # print(label,amount)
            # p(y)
            self.prior[label] = (amount + self.lamb) / (m + k * self.lamb)  # 计算平滑处理后的先验概率
        for feature in range(n):  # 遍历每个特征
            self.conditional[feature] = {}
            values = np.unique(features[:, feature])
            # print(values)
            for value in values:  # 遍历每个特征值
                # print(value)
                # print(1111111)
                self.conditional[feature][value] = {}
                for label, amount in labels.items():  # 遍历每种类别
                    feature_label = features[target[:, 0] == label, :]  # 截取该类别的数据集
                    # print(feature_label)
                    c_label = Counter(feature_label[:, feature].flatten().tolist())  # 计算该类别下各特征值出现的次数
                    self.conditional[feature][value][label] = (c_label.get(value, 0) + self.lamb) / \
                                                              (amount + len(values) * self.lamb)  # 计算平滑处理后的条件概率
        return

    def predict(self, features):
        """预测单个样本"""
        best_poster, best_label = -np.inf, -1
        for label in self.prior:
            poster = np.log(self.prior[label])  # 初始化后验概率为先验概率,同时把连乘换成取对数相加，防止下溢（即太多小于1的数相乘，结果会变成0）
            for feature in range(features.shape[0]):
                poster += np.log(self.conditional[feature][features[feature]][label])
            if poster > best_poster:  # 获取后验概率最大的类别
                best_poster = poster
                best_label = label
        return best_label


def test():
    # dataset = datasets.load_iris()  # 鸢尾花数据集
    # dataset = np.concatenate((dataset['data'], dataset['target'].reshape(-1, 1)), axis=1)  # 组合数据
    # np.random.shuffle(dataset)  # 打乱数据
    # features = dataset[:, :-1]
    # target = dataset[:, -1:]
    # print(features,type(features))
    # print(target,type(target))
    csv_file = open('data.csv')  # 打开文件
    csv_reader_lines = csv.reader(csv_file)  # 用csv.reader读文件
    dataset = []
    for one_line in csv_reader_lines:
        dataset.append(one_line)  # 逐行将读到的文件存入python的列表
    dataset = np.array(dataset)  # 将python列表转化为ndarray
    # print(dataset,type(dataset))
    dataset = dataset.astype('float64')
    features = dataset[:, :-1]
    target = dataset[:, -1:]
    # print(features)
    # print(target)
    test_PCA(features,target)
    plot_PCA(features,target)
    nb = NaiveBayes()
    nb.training(features, target)
    prediction = []
    for features in features:
        prediction.append(nb.predict(features))
    correct = [1 if a == b else 0 for a, b in zip(prediction, target)]
    print(correct.count(1) / len(correct))  # 计算准确率

# def load_data():
#     iris=datasets.load_iris()
#     return iris.data,iris.target

def test_PCA(*data):
    X,y=data
    pca=decomposition.PCA(n_components=None)
    pca.fit(X)
    print('explained variance ratio : %s'%str(pca.explained_variance_ratio_))

# X,y=load_data()
# test_PCA(X,y)

def plot_PCA(*data):
    X,y=data
    pca=decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r=pca.transform(X)
    # print(X_r,np.shape(X_r))
    # print(y,np.shape(y))
    Y = y.ravel()
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
    l = ["堆肥","回收","焚烧","填埋"]
    i = 0
    for label,color in zip(np.unique(Y),colors):
        position=Y==label
        # print(position)
        ax.scatter(X_r[position,0],X_r[position,1],label="%s"%l[i],color=color)
        i += 1
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best")
    ax.set_title("垃圾分类图")
    plt.show()
# plot_PCA(X,y)
if __name__ == '__main__':
    test()


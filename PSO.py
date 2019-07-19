#算法流程
'''
Initial：
	初始化粒子群体（群体规模为n），包括随机位置和速度。
Evaluation：
	根据fitness function ，评价每个粒子的适应度。
Find the Pbest：
   对每个粒子，将其当前适应值与其个体历史最佳位置（pbest）对应的适应值做比较，如果当前的适应值更高，则将用当前位置更新历史最佳位置pbest。
Find the Gbest：
	对每个粒子，将其当前适应值与全局最佳位置（gbest）对应的适应值做比较，如果当前的适应值更高，则将用当前粒子的位置更新全局最佳位置gbest。
Update the Velocity：
	根据公式更新每个粒子的速度与位置。
如未满足结束条件，则返回步骤2     (最大迭代步数;可接受的满意解)
    通常算法达到最大迭代次数     或者最佳适应度值的增量小于某个给定的阈值时算法停止。
'''
import numpy as np
import math
import matplotlib.pyplot as plt
import map
import m
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import math

class PSO:
    velocity = []   #速度
    position = []   #位置
    fitvalue = []   #适应度值
    dimension = 2   #维度
    pbest = []  #个体最好位置
    gbest = []  #种群最好位置
    low = 0     #下界
    up = 0  #上界
    popNum = 100    #粒子个数
    c1 = 2.0    #自我认知
    c2 = 2.1    #社会经验
    w = 0      #惯性权重   wmax = 0.9  wmin = 0.4
    wmax = 0.9
    wmin = 0.4
    iteration = 500   #最大迭代次数

    def __init__(self,dimension = 2, low = 0, up = 0, popNum = 50, c1 =2.0, c2 = 2.1, wmax = 0.9, wmin = 0.4, iteration = 200):    #构造方法
        self.dimension = dimension
        self.low = low
        self.up = up
        self.popNum = popNum
        self.c1 = c1
        self.c2 = c2
        self.wmax = wmax
        self.wmin = wmin
        self.iteration = iteration

    def __initial__(self):    #初始化粒子群位置和速度
        #print(1)
        for i in range(self.popNum):
            list = []
            list1 = []
            for j in range(self.dimension):
                list.extend([np.random.uniform(self.low, self.up)])
                list1.extend([np.random.uniform(0.15*self.low, 0.15*self.up)])
            self.position.append(list)
            self.velocity.append(list1)
        #print(self.position)    #[[x1,x2][x1,x2]]
        #print(self.velocity)

    def __fitValue__(self,fun_value):    #计算适应度值
        #print(2)
        self.fitvalue = []
        k  = 10
        maximum = max(fun_value)
        minimum = min(fun_value)
        for i in range(len(fun_value)):
            self.fitvalue.append((fun_value[i] - minimum + k)/(maximum - minimum + k))
        #print(self.fitvalue)

    def __findPbest__(self):    #找到个体最优位置
        #print(3)
        if self.pbest== []:
            self.pbest = self.position
        else:
            for i in range(self.popNum):
                if  funValue(0,self.pbest[i])< funValue(0, self.position[i]):
                    self.pbest[i] = self.position[i]
        #print(self.pbest)

    def __findGbest__(self):    #找到群体最优位置
        #print(4)
        Gbest = []
        Gbest = self.position[np.argmax(self.fitvalue)]
        if self.gbest == []:
            self.gbest = Gbest
        else:
            if funValue(0,self.gbest)< funValue(0, Gbest):
                self.gbest = Gbest
        #print(self.gbest)

    def __update__(self, iter):    #更新
        #print(5)
        self.w = self.wmax - (self.wmax - self.wmin)*iter/self.iteration    #更新惯性权重
        #print(self.w)
        for i in range(self.popNum):
            for j in range(self.dimension):
                self.velocity[i][j] = self.w * self.velocity[i][j] + self.c1*np.random.uniform(0,1)*(self.pbest[i][j]-self.position[i][j]) \
                                      + self.c2*np.random.uniform(0,1)*(self.gbest[j]-self.position[i][j])
                self.position[i][j] = self.position[i][j] + self.velocity[i][j]
                if self.position[i][j] <= self.low:
                    self.position[i][j] = self.low
                if self.position[i][j] >= self.up:
                    self.position[i][j] = self.up

#计算某一个函数的函数值

def funValue(population,position):
    fun_value = []
    if population == 0:
        # fun_value.append(100 * pow((position[1] - pow(position[0], 2)), 2) + pow(1 - position[0], 2))
        fun_value.append(-sum(pow(map.Xd[:,0]-position[0],2))-sum(pow(map.Xd[:,1]-position[1],2)))

    for i in range(population):
        # fun_value.append(100*pow((position[i][1]-pow(position[i][0],2)),2)+pow(1-position[i][0],2))
        fun_value.append(-sum(pow(map.Xd[:, 0] - position[i][0], 2)) - sum(pow(map.Xd[:, 1] - position[i][1], 2)))
    #print(fun_value)
    return fun_value




if __name__ == '__main__':    #主函数

    pso = PSO(dimension = 2, low = -30, up = 30, popNum = 100, iteration = 200)
    pso.__initial__()      #粒子个数
    x1, x2, y = [], [], []
    i = 0
    for i in range(pso.iteration):
        fun_value = funValue(pso.popNum, pso.position)
        pso.__fitValue__(fun_value)
        pso.__findPbest__()
        pso.__findGbest__()
        y.extend(funValue(0,pso.gbest))
        x1.append(pso.gbest[0])
        x2.append(pso.gbest[1])

        pso.__update__(i)

    print(x1)
    print(x2)
    print(y)
    plt.figure(1)
    plt.title("PSO")
    plt.xlabel("迭代次数")
    plt.ylabel("函数值")
    plt.plot(y)
    plt.savefig("PSO.png")
    if pso.dimension == 2:
        plt.figure(2)
        plt.title("PSO")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.plot(x1, x2,"*b")
        plt.savefig("PSO.jpg")

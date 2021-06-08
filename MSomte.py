from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from numpy import *


class MSmote(object):

    def __init__(self, N=50, k=5, r=2):
        # 初始化self.N, self.k, self.r, self.newindex
        self.N = N
        self.k = k
        # self.r是距离决定因子
        self.r = r
        # self.newindex用于记录SMOTE算法已合成的样本个数
        self.newindex = 0

    # 构建训练函数
    def fit(self, samples):
        # 初始化self.samples, self.T, self.numattrs
        self.samples = samples
        # self.T是少数类样本个数，self.numattrs是少数类样本的特征个数
        self.T, self.numattrs = self.samples.shape

        # 查看N%是否小于100%
        if (self.N < 100):
            # 如果是，随机抽取N*T/100个样本，作为新的少数类样本
            np.random.shuffle(self.samples)
            self.T = int(self.N * self.T / 100)
            self.samples = self.samples[0:self.T, :]
            # N%变成100%
            self.N = 100

        # 查看从T是否不大于近邻数k
        if (self.T <= self.k):
            # 若是，k更新为T-1
            self.k = self.T - 1

        # 令N是100的倍数
        N = int(self.N / 100)
        # 创建保存合成样本的数组
        self.synthetic = np.zeros((self.T * N, self.numattrs))

        # 调用并设置k近邻函数
        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(self.samples)

        # 对所有输入样本做循环
        for i in range(len(self.samples)):
            mid =samples.iloc[i]
            samples_np =np.array(samples.iloc[i])
            # 调用kneighbors方法搜索k近
            nnarray = neighbors.kneighbors(samples_np.reshape((1, -1)),
                                           return_distance=False)[0][1:]


            # 把N,i,nnarray输入样本合成函数self._populate
            self.__populate(N, i, nnarray, samples)

        # 最后返回合成样本self.synthetic
        return self.synthetic

    # 构建合成样本函数
    def __populate(self, N, i, nnarray, samples):
        # 按照倍数N做循环
        for j in range(N):
            # attrs用于保存合成样本的特征
            attrs = []
            # 随机抽取1～k之间的一个整数，即选择k近邻中的一个样本用于合成数据
            nn = random.randint(0, self.k - 1)

            # 计算差值
            diff = samples.iloc[nnarray[nn]] - samples.iloc[i]

            dis_nn = self.__manhattan_distance(samples, samples.iloc[nnarray[nn]])
            dis_i = self.__manhattan_distance(samples, samples.iloc[i])
            #print(dis_nn, dis_i)
            # 随机生成一个0～1之间的数
            if (dis_nn > dis_i):
                gap = random.uniform(0.5, 1)
            else:
                gap = random.uniform(0, 0.5)
            # 合成的新样本放入数组self.synthetic
            self.synthetic[self.newindex] = self.samples.iloc[i] + gap * diff

            # self.newindex加1， 表示已合成的样本又多了1个
            self.newindex += 1

    def __manhattan_distance(self, x, narray):
        x = x.drop(['protocol_type', 'land','wrong_fragment','urgent','num_outbound_cmds','is_host_login','label'], axis=1)
        x_mean = np.mean(np.array(x), axis=0)
        x = np.array(x)
        xT = x.T
        D = np.cov(xT)
        invD = np.linalg.inv(D)
        narray_ch = narray.drop(['protocol_type', 'land','wrong_fragment','urgent','num_outbound_cmds','is_host_login','label'], axis=0)
        tp = narray_ch - x_mean
        dis = np.sqrt(dot(dot(tp, invD), tp.T))
        return dis
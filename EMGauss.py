# 求解高斯混合模型的EM算法
import math
import numpy as np

class Em:
    def __init__(self, a, K, y):
        """

        :param a: list
        :param K: int 模型个数
        :param y: list 观测值
        :param mean: list 正态分布 平均值集合 len = K
        :param squre: list 正态分布 方差集合 len = K
        :param r: 矩阵 迭代值 规模为 K*N
        """
        self.a = a
        self.K = K
        self.mean = [2 for _ in range(K)]
        self.y = y
        self.N = len(self.y)
        self.squre = [3 for _ in range(K)]
        self.r = [[1] for _ in range(self.N)]
        for j in range(self.N):
            self.r[j] = [1 for _ in range(self.K)]

    def cal_gauss(self, j, k):
        result = self.a[k] / (math.sqrt(2 * math.pi) * self.squre[k]) * math.exp(
            -((self.y[j] - self.mean[k]) ** 2) / (2 * self.squre[k] ** 2))
        return result

    def section_e(self):
        for j in range(self.N):
            sum_ = 0
            for k in range(self.K):
                sum_ += self.cal_gauss(j, k)
            self.r[j][k] = self.cal_gauss(j, k) / sum_

    def section_m(self):
        self.section_e()
        for k in range(self.K):
            segma_rjk = 0
            segma_rjk_yj = 0
            segma_rjk_yj_meank = 0
            for j in range(self.N):
                segma_rjk += self.r[j][k]
                segma_rjk_yj += self.r[j][k] * self.y[j]
                segma_rjk_yj_meank += self.r[j][k] * (self.y[k] - self.mean[k]) ** 2
            self.mean[k] = segma_rjk_yj / segma_rjk
            self.squre[k] = segma_rjk_yj_meank / segma_rjk
            self.a[k] = segma_rjk / self.N

    def fit(self,iter=100):
        for i in range(iter):
            self.section_e()
            self.section_m()
            print(self.mean)
            print(self.squre)
            print(self.a)


Y = [-64, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]
K = 2
a = [0.5, 0.5]
em = Em(a, K, Y)
em.fit()


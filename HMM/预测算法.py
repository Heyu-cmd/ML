# 维特比算法实现
# by heyup
import numpy as np


class HMM_predict:
    def __init__(self):
        self.O_vector = None  # 观测序列
        self.A_matrix = None  # 状态转移矩阵
        self.B_matrix = None  # 观测概率分布
        self.pi_vector = None  # 初始概率分布
        self.t_len = None  # t = 0,1,2,3,...,T-1 总长度为T
        self.i_len = None  # 状态数量
        self.dp = None  # 动态规划矩阵 定义时刻t状态为i的所有单个路径中概率最大值
        self.node = None  # 时刻t 状态为i 的所有单个路径中概率最大的路径的第t-1个结点

    def args_init(self, A_matrix, B_matrix, pi_vector, O_vector):
        self.A_matrix = A_matrix
        self.B_matrix = B_matrix
        self.pi_vector = pi_vector
        self.O_vector = O_vector
        self.t_len = len(O_vector)
        self.i_len = len(pi_vector)
        self.dp = [[0] for _ in range(self.t_len)]
        for t in range(self.t_len):
            self.dp[t] = [0 for _ in range(self.i_len)]
        self.node = [[0] for _ in range(self.t_len)]
        for t in range(self.t_len):
            self.node[t] = [0 for _ in range(self.i_len)]

    def first_cal(self):
        o1 = self.O_vector[0]
        for i in range(self.i_len):
            self.dp[0][i] = self.pi_vector[i] * self.B_matrix[i][o1]
            self.node[0][i] = 0

    def max_dp(self, t, i):
        dict_ = {}
        for j in range(self.i_len):
            dict_ = {j: self.dp[t - 1][j] * self.A_matrix[j][i]}
        best_ = max(dict_.items(), key=lambda x: x[1])
        return best_[1], best_[0]

    def fit(self, A_matrix, B_matrix, pi_vector, O_vector):
        self.args_init(A_matrix, B_matrix, pi_vector, O_vector)
        self.first_cal()
        for t in range(1, self.t_len):
            for i in range(self.i_len):
                value, node = self.max_dp(t, i)
                self.dp[t][i] = value * self.B_matrix[i][self.O_vector[t]]
                self.node[t][i] = node
        t = self.t_len - 1
        node_list = [0 for _ in range(self.t_len)]
        # 最后时刻最大概率对应的结点为
        maxindex = np.argmax(self.dp[t])
        node_list[t] = maxindex
        while t > 0:
            maxindex = self.node[t][maxindex]
            t = t - 1
            node_list[t] = maxindex
        return node_list


if __name__ == '__main__':
    forward = HMM_predict()
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    A = np.array(A)
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    B = np.array(B)
    pi = [0.2, 0.4, 0.4]
    pi = np.array(pi)
    O = [0, 1, 0]
    O = np.array(O)
    # 返回的值是索引，实际值应该加1
    print(forward.fit(A, B, pi, O))

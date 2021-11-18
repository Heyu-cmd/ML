# 前向算法计算概率
# 输入HMM模型参数，观测序列O
import numpy as np


class Forword:
    def __init__(self):
        pass

    def args_init(self, a_matrix, pi_matrix, b_matrix, O_vector):
        self.i = a_matrix.shape[0]
        self.t_len = len(O_vector)
        self.alpha = [[0] for _ in range(self.t_len)]
        for t in range(self.t_len):
            self.alpha[t] = [0 for _ in range(self.i)]

    def first_value(self, pi_vector, bi_matrix, O_vector):
        # 观测序列的第一个
        o1 = O_vector[0]
        for i in range(self.i):
            self.alpha[i][0] = pi_vector[i] * bi_matrix[i][o1]

    def cal_sum(self, t, i, a_matrix):
        sum_ = 0
        for j in range(self.i):
            sum_ += self.alpha[j][t] * a_matrix[j][i]
        return sum_

    def fit(self, a_matrix, pi_vector, b_matrix, O_vector):
        """

        :param a_matrix: 状态转移矩阵
        :param pi_vector: 初始概率分布
        :param b_matrix: 观测概率分布
        :param O_vector: 观测序列
        :return:
        """
        self.args_init(a_matrix, pi_vector, b_matrix, O_vector)
        self.first_value(pi_vector, b_matrix, O_vector)
        for t in range(self.t_len - 1):
            for i in range(self.i):
                self.alpha[i][t + 1] = self.cal_sum(t, i, a_matrix) * b_matrix[i][O_vector[t + 1]]
        P = 0
        for i in range(self.i):
            P += self.alpha[i][-1]
        return P


if __name__ == '__main__':
    forward = Forword()
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    A = np.array(A)
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    B = np.array(B)
    pi = [0.2, 0.4, 0.4]
    pi = np.array(pi)
    O = [0, 1, 0]
    O = np.array(O)
    print(forward.fit(A,pi,B,O))

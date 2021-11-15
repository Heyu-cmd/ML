# SMO求解SVM
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    """

    :return: X-二维 y-一维
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    df['label'] = iris.target
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for d in range(len(data)):
        if data[d, -1] == 0:
            data[d, -1] = -1
    return data[:, :], data[:, -1]


class SVM:
    def __init__(self, max_iter=1000, kernel="linear", p=2):
        """

        :param max_iter: 最大跌打次数
        :param kernel: 默认是线性SVM
        """
        self.max_iter = max_iter
        self.kernel = kernel
        # 多项式核函数默认次数为2
        self.p = p

    def init_args(self, features, label, c=1.0):
        """
        线性SVM：没有松弛
        :param features: 训练集 x
        :param label: 训练集 y
        :return:
        """
        # m：样本数  n：特征数
        self.m, self.n = features.shape
        self.X = features
        self.Y = label
        # alpha 的个数等于y的个数
        self.alpha = np.ones(self.m)
        self.b = 0.0
        # E的个数 等于y的个数
        # 把Ei保存到self._Ei
        self.E = [self._E(i) for i in range(self.m)]
        # 加入松弛遍历
        self.C = c

    def _E(self, i):
        """
        Ei = g(xi)-yi
        :param i: 下标
        :return: 预测结果和真是结果的差值
        """
        return self.gx(i) - self.Y[i]

    def gx(self, j):
        """

        :param i: 输入的是下标
        :return: 输出预测结果
        """
        sum_ = self.b
        for i in range(self.m):
            sum_ += self.alpha[i] * self.Y[i] * self.kernel_function(self.X[i], self.X[j])
        return sum_

    def kernel_function(self, xi, xj):
        """

        :param xi: X的第i行数据
        :param xj: X的第j行数据
        :return: 计算结果
        """
        if self.kernel == "linear":
            return sum(xi[_] * xj[_] for _ in range(self.n))
        elif self.kernel == "poly":
            # 多项式核函数
            return (sum(xi[_] * xj[_] for _ in range(self.n)) + 1) ** self.p

    def KKT_judge(self, i):
        yg = self.Y[i] * self.gx(i)
        ai = self.alpha[i]
        if ai == 0:
            return yg >= 1
        elif 0 < ai < self.C:
            return yg == 1
        else:
            return yg <= 1

    def select_alpha_1(self):
        """
        选择第一个alpha
        :return:i-第一个alpha的下标，j第二个alpha的下标
        """
        # 1 第一个变量的选择，外层循环，首先遍历所有alpha大于0小于C的点,检验是否都满足KKT条件
        # 检查在边界上的点是否满足kkt条件
        border_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C and not self.KKT_judge(i)]
        if border_list:
            # 如果存在违反KKT条件的边界点
            return border_list[0]
        else:
            for j in range(self.m):
                if self.alpha[j] == 0 or self.alpha[j] == self.C:
                    if not self.KKT_judge(j):
                        return j

    def select_alpha(self):
        """

        :return: second-对应E2的下标
        """
        first = self.select_alpha_1()
        E1 = self.E[first]
        for i in range(self.m):
            if i == first:
                continue
            if E1 >= 0:
                second = min(range(self.m), key=lambda x: self.E[x] and x != first)
            else:
                second = max(range(self.m), key=lambda x: self.E[x] and x != first)
        return first, second

    def alpha_judge(self, alpha_unc, L, H):
        if alpha_unc > H:
            return H
        elif L <= alpha_unc <= H:
            return alpha_unc
        else:
            return L
    def judge_iter(self):
        sum_ = 0
        for i in range(self.m):
            if self.alpha[i] < 0 or self.alpha[i] >self.C:
                return False
            sum_ += self.alpha[i]*self.Y[i]
        if sum_ == 0:
            return True
        else:
            return False
    def fit(self, train_x, train_y):
        self.init_args(train_x, train_y)

        for k in range(self.max_iter):
            i1, i2 = self.select_alpha()
            alpha2_old = self.alpha[i2]
            alpha1_old = self.alpha[i1]
            # 新的alpha需要满足条件 L<= ALPHA <= H
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
            else:
                L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
                H = min(self.C, self.alpha[i2] + self.alpha[i1])
            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta = K11+K22-2K12
            a = self.X[i1]
            b = self.X[i2]
            eta = self.kernel_function(a, a) + self.kernel_function(b, b) - 2 * self.kernel_function(a, b)
            alpha_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = self.alpha_judge(alpha_new_unc, L, H)
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)
            b1_new = -E1 - self.Y[i1] * self.kernel_function(a, a) * (alpha1_new - alpha1_old) - self.Y[
                i2] * self.kernel_function(b, a) * (alpha2_new - alpha2_old) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel_function(a, b) * (alpha1_new - alpha1_old) - self.Y[
                i2] * self.kernel_function(b, b) * (alpha2_new - alpha2_old) + self.b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2
            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
            # 如果所有的alpha满足kkt条件，迭代结束
            if self.judge_iter():
                print("SVM:kernel-{},iter-{}".format(self.kernel, k+1))
                return
        print("SVM:kernel-{},iter-{}".format(self.kernel, self.max_iter))


    def predict(self, test):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel_function(test, self.X[i])
        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)


if __name__ == '__main__':
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    svm = SVM()
    svm.fit(X_train, y_train)
    print(svm.score(X_test, y_test))

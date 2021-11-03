"""
	感知机(对偶形式)：实现书本P40页 例题2.1
"""

import numpy as np
import matplotlib.pyplot as plt


class MyPerceptron():
    def __init__(self):
        # 因为w的维度与x一致，所以先不定义为固定维度
        self.a = None
        self.w = None
        self.b = 0
        self.lr = 1

    def fit(self, x_train, y_train, Gram_Matrix):
        self.a = np.zeros(x_train.shape[0]) # a是x的行数，表示一共有多少条数据
        i = 0
        while i < x_train.shape[0]:
            x = x_train[i]
            y = y_train[i]
            # 若为误分类点则更新参数，重新遍历样本集
            print(self.a)
            print(y_train)
            print(self.a * y_train)
            print(Gram_Matrix[:, i])
            print("======")
            if y * (np.sum(self.a * y_train * Gram_Matrix[:, i]) + self.b) <= 0:
                self.a[i] = self.a[i] + self.lr
                self.b = self.b + self.lr * y
                i = 0
            else:
                i += 1

        # 因为看公式可以知道y_train与self.a是对应位置相乘，他们与x_train是矩阵乘法
        # 所以将x_train转换为矩阵（array中的*代表对应位置相乘，而mat中的*为矩阵乘法）
        self.w = y_train * self.a * np.mat(x_train)

        # 这里修改下w维度，为了和原始形式程序中的draw函数统一
        self.w = np.squeeze(np.array(self.w))


def draw(X, w, b):
    # 生产分离超平面上的两点
    X_new = np.array([0, 6])
    y_predict = -(b + w[0] * X_new) / w[1]
    # 绘制训练数据集的散点图
    plt.plot(X[:2, 0], X[:2, 1], "g*", label="1")
    plt.plot(X[2:, 0], X[2:, 1], "rx", label="-1")
    # 绘制分离超平面
    plt.plot(X_new, y_predict, "b-")
    # 设置两坐标轴起止值
    plt.axis([0, 6, 0, 6])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def main():
    x_train = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    # 生成Gram矩阵
    Gram_Matrix = np.zeros(shape=(3, 3))
    for i in range(3):
        for j in range(3):
            Gram_Matrix[i][j] = np.dot(x_train[i], x_train[j].T)

    Perceptron = MyPerceptron()
    Perceptron.fit(x_train, y, Gram_Matrix)
    draw(x_train, Perceptron.w, Perceptron.b)


if __name__ == "__main__":
    main()

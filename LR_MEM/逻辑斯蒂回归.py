# Logistic Regression
# 《统计学习方法》采用的是极大似然估计，使用梯度下降和拟牛顿法获取最优值
# 这里采用的是损失函数最小化
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math


class LogisticRegression:
    def __init__(self, max_iter=100, learn_rate=0.01):
        # 梯度下降法的最大迭代次数和学习率
        self.max_iter = max_iter
        self.learning_rate = learn_rate

    def sigmoid(self, x):
        # logistic分布函数
        return 1 / (1 + math.exp(-x))
    def data_matrix(self, X):
        data_mat = []
        for d in X:
            # 将wx+b 转化为 wx形式
            data_mat.append([1.0, *d])
        return data_mat

    def fit(self, X, y):
        data_mat = self.data_matrix(X)  # m*n
        # weights即需要优化的参数，初始化为0
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                # 计算 1 / （1+exp（w·x））
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                # 计算损失
                error = y[i] - result
                # 计算XT（y-H）损失函数求对w求偏导获得
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])

        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    def predict(self, test_x):
        test = [1.0]
        for x in test_x:
            test.append(x)
        test = np.array(test)
        p1 = self.sigmoid(np.dot(test, self.weights))
        if p1 > 0.5:
            return 1
        else:
            return 0


def create_data():
    iris = load_iris()
    X, y = np.array(iris.data), np.array(iris.target)
    return X[:100, 0:2], y[:100]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr = LogisticRegression()
lr.fit(X_train, y_train)
length = len(X_test)
right = 0
for x in range(length):
    class_ = lr.predict(X_test[x])
    if class_ == y_test[x]:
        right += 1
# 准确率将近 97%
print(right/length)

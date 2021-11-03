from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['label'] = iris.target  # iris['target']

datas = np.array(df.loc[:151,])
x, y = datas[:, 0:4], datas[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])  # 把标签y当中的值全编程-1或者+1


class Perceptron:
    def __init__(self, x_train, y_train, rate):
        """

        :param x_train: n 维向量
        :param y_train: 1 or -1
        :param rate: 学习率
        """
        self.x_train = x_train
        self.y_train = y_train
        self.w = [0] * len(self.x_train[0]) # w也是n维向量。
        self.b = 0
        self.rate = rate

    def fit(self):
        i = 0
        while i < len(self.x_train):
            xi = self.x_train[i]
            yi = self.y_train[i]
            if yi * (np.dot(self.w, xi) + self.b) <= 0:
                # 如果存在误分类，进行梯度下降优化
                self.w = self.w + self.rate * np.dot(xi, yi)
                self.b = self.b + self.rate * yi
                # 初始化i
                i = 0
            else:
                i += 1
                print(i)


perceptron = Perceptron(x, y, rate=0.6)
perceptron.fit()
print(perceptron.w)
print(perceptron.b)


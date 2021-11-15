import numpy as np
import pandas as pd
from numpy import shape,power
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Cart:
    def __init__(self):
        pass

    def binSplitDataSet(self, dataset, feature, value):
        """
        切分数据集
        :param dataset:待切分数据集
        :param feature:待切分特征序号
        :param value:切分值
        :return:切分数据集1 切分数据集2
        """
        # np.nonzero: 输出数据集中所有非0数据的位置，下标是从0开始
        # np.nonzero(dataset[:, feature] > value)：第feature个特征对应的值中，大于value的数值的位置
        # 选择第feature个特征值大于value的数据
        mat0 = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
        mat1 = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
        return mat0, mat1

    def cal_err(self, dataset):
        # 计算平方误差
        err = np.var(dataset[:, -1]) * np.shape(dataset)[0]
        print("---------")
        print(np.var(dataset[:, -1]))
        print(np.shape(dataset)[0])
        return err

    def reg_leaf(self, dataset):
        # 生成叶子节点，为当前特征值的期望
        return np.mean(dataset[:, -1])

    def select_best_split(self, dataset, ops=(1, 4)):
        # 获得相关参数中的最大样本数和最小误差效果 提升值
        tols = ops[0]
        toln = ops[1]
        n = shape(dataset)[1]
        # 如果 所有的样本点的值一致，建立叶子节点

        if len(set(dataset[:, -1].T.tolist())) == 1:
            return None, self.reg_leaf(dataset)
        # 不划分时的误差
        gap_no_split = self.cal_err(dataset)
        # 最小误差
        best_gap = np.inf
        # 划分后的特征下标和特征值
        index = 0
        value = 0
        # 选择最优特征
        for f_index in range(n - 1):
            # 对于某个特征的所有特征值
            for f_val in set(dataset[:, f_index]):
                # 进行划分
                mat0, mat1 = self.binSplitDataSet(dataset, f_index, f_val)
                # 如果划分后某个子集不满足条件
                if shape(mat0)[0] < toln or shape(mat1)[0] < toln: continue
                # 当前划分的误差
                gap_split = self.cal_err(mat0) + self.cal_err(mat1)
                if gap_split < best_gap:
                    index = f_index
                    value = f_val
                    best_gap = gap_split
        # 如果当前划分所提升的效果不满意，建立叶子节点
        if gap_no_split - best_gap < tols:
            return None, self.reg_leaf(dataset)
        # 按照最优方式进行划分
        setl, selr = self.binSplitDataSet(dataset, index, value)
        # 如果划分后某个子集的个数不达标
        if shape(setl)[0] < toln or shape(selr)[0] < toln:
            return None, self.reg_leaf(dataset)
        return index, value

    def creat_tree(self, dataset, ops=(1, 4)):
        # 选择最优划分特征，以及对应的特征值
        index, value = self.select_best_split(dataset, ops)
        # 如果index为None，返回叶节点
        if index == None:
            return value
        regTree = {}
        regTree['index'] = index
        regTree['value'] = value
        lset, rset = self.binSplitDataSet(dataset, index, value)
        regTree['left'] = self.creat_tree(lset, ops)
        regTree['right'] = self.creat_tree(rset, ops)

        return regTree

    def is_tree(self, tree):
        # 判断是不是树
        if type(tree) == 'dict':
            return True

    def get_mean(self, tree):
        # 坍塌处理
        if self.is_tree(tree['left']): tree['left'] = self.get_mean(tree['left'])
        if self.is_tree(tree['right']): tree['right'] = self.get_mean(tree['right'])
        return (tree['left'] + tree['right']) / 2.0

    def prune(self, tree, testData):
        """
        后剪值
        :param tree: 处理对象
        :param testData:测试数据集
        :return:剪枝后的树
        """
        if shape(testData)[0] == 0:
            # 无测试集则坍塌此树
            return self.get_mean(tree)
        # 如果左子集 或者 右子集是树
        if self.is_tree(tree['left']) or self.is_tree(tree['right']):
            # 划分测试集
            mat0, mat1 = self.binSplitDataSet(testData, tree['index'], tree['value'])
        # 在新树测试集上递归剪枝
        if self.is_tree(tree['left']): self.prune(tree['left'],mat0)
        if self.is_tree(tree['right']): self.prune(tree['right'],mat1)

        # 如果两个子集都是叶子节点的话，则在进行误差评估后判断是否合并
        if not self.is_tree(tree['left']) and not self.is_tree(tree['right']):
            lSet, rSet = self.binSplitDataSet(testData, tree['index'], tree['value'])
            errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
            treeMean = (tree['left'] + tree['right']) / 2.0
            errorMerge = sum(power(testData[:, -1] - treeMean, 2))
            if errorMerge < errorNoMerge:
                return treeMean
            else:
                return tree
        else:
            return tree

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
    return data[:, :]


if __name__ == '__main__':
    train = create_data()
    cart = Cart()
    print(cart.creat_tree(train))

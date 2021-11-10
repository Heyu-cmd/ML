# C4.5 决策树实现及其剪枝
from math import log
import pandas as pd
import numpy as np


class Tree:
    def __init__(self, label, feature=None, feature_index=None):
        # 定义每个节点的属性
        self.label = label  # 标签值
        self.tree = {}  # 子树
        self.feature = feature  # 被分割的属性，叶子节点没有该属性
        self.feature_index = feature_index
        self.result = {'label:': self.label, 'feature': self.feature,'feature_index':self.feature_index, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)
    def add_node(self, val, node):
        self.tree[val] = node
    def predict(self,data):
        if self.feature_index == None:
            return self.label
        return self.tree[data[self.feature_index]].predict(data)

class C45:
    def __init__(self, eta=0):
        self.decision_tree = None
        self.eta = eta

    def entropy(self, dataframe):
        # 计算数据集的熵
        data_length = len(dataframe)
        mark_dict = {}
        for i in range(data_length):
            label = dataframe[i][-1]
            if label not in mark_dict:
                mark_dict[label] = 1
            else:
                mark_dict[label] += 1
        ent = -sum((p / data_length) * log(p / data_length, 2) for p in mark_dict.values())
        return ent

    def condition_entropy(self, dataframe, dim):
        # 1.根据维度划分数据集
        data_len = len(dataframe)
        dim_dict = {}
        for i in range(data_len):
            dim_feature = dataframe[i][dim]
            if dim_feature not in dim_dict:
                dim_dict[dim_feature] = []
            dim_dict[dim_feature].append(dataframe[i])
        cond_ent = sum(len(p) / data_len * self.entropy(p) for p in dim_dict.values())
        ent_dim = -sum(len(p) / data_len * log(len(p) / data_len, 2) for p in dim_dict.values())
        return cond_ent, ent_dim

    def rate_info_gain(self, dataframe, dim):
        ent_con, ent_dim = self.condition_entropy(dataframe, dim)
        info_gain = self.entropy(dataframe) - ent_con
        ent_rate = info_gain / ent_dim
        return ent_rate

    def best_dim(self, array):
        count = len(array[0]) - 1
        dim_info_gain = []
        for i in range(count):
            dim_info_gain.append((i, self.rate_info_gain(array, i)))
        info_gain = max(dim_info_gain, key=lambda x: x[1])
        return info_gain[0], info_gain[1]

    def train(self, dataframe):
        train_x, train_y, features = dataframe.iloc[:, :-1], dataframe.iloc[:, -1], dataframe.columns[:-1]
        max_label = train_y.value_counts().sort_values(ascending=False).index[0]
        if len(train_y.value_counts()) == 1:
            # 如果所有实例都属于同一类，则将y作为根节点的label
            return Tree(label=train_y.iloc[0])
        if len(features) == 0:
            # 如果特征为空集，则将数量最多的类别作为根节点的label
            return Tree(label=max_label)
        # 计算信息增益比最大的特征
        max_dim, info_gain = self.best_dim(np.array(dataframe))
        max_dim_name = features[max_dim]
        # 如果信息增益比小于阈值
        if info_gain < self.eta:
            return Tree(label=max_label)
        # 如果信息增益比大于阈值
        tree = Tree(label=max_label, feature=max_dim_name, feature_index=max_dim)
        this_dim_data = dataframe[max_dim_name].value_counts().index
        for d in this_dim_data:
            sub_dataframe = dataframe.loc[dataframe[max_dim_name] == d].drop(max_dim_name, axis=1)
            tree.add_node(d, self.train(sub_dataframe))
        return tree

    def fit(self, dataframe):
        self.decision_tree = self.train(dataframe)

    def predict(self,data):
        if self.decision_tree == None:
            print("Please fit")
            return
        else:
            return self.decision_tree.predict(data)
    def pruning(self):
        # 1、 计算每个节点的经验熵
        # 计算剪枝前和剪枝后的损失函数变化
        pass
if __name__ == '__main__':
    dataSet = [['青年', '否', '否', '一般', '拒绝'],
               ['青年', '否', '否', '好', '拒绝'],
               ['青年', '是', '否', '好', '同意'],
               ['青年', '是', '是', '一般', '同意'],
               ['青年', '否', '否', '一般', '拒绝'],
               ['中年', '否', '否', '一般', '拒绝'],
               ['中年', '否', '否', '好', '拒绝'],
               ['中年', '是', '是', '好', '同意'],
               ['中年', '否', '是', '非常好', '同意'],
               ['中年', '否', '是', '非常好', '同意'],
               ['老年', '否', '是', '非常好', '同意'],
               ['老年', '否', '是', '好', '同意'],
               ['老年', '是', '否', '好', '同意'],
               ['老年', '是', '否', '非常好', '同意'],
               ['老年', '否', '否', '一般', '拒绝'],
               ]
    featureName = ['年龄', '有工作', '有房子', '信贷情况', '类别']
    data_df = pd.DataFrame(dataSet, columns=featureName)
    alo = C45()
    alo.fit(data_df)
    # print(alo.predict(['青年', '否', '否', '一般']))
    print(alo.pruning())

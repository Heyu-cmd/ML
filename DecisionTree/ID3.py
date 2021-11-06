import pandas as pd
import numpy as np
from math import log

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
featureName = ['年龄', '有工作', '有房子', '信贷情况','类别']
data_df = pd.DataFrame(dataSet, columns=featureName)


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        """

        :param root: 是否只有根节点
        :param label: 类别
        :param feature: 特征名
        """
        self.root = root
        self.label = label
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        print(self.tree)
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class Id3:
    def __init__(self, epsilon=0.1):

        self.epsilon = epsilon
    def entropy(self, dataset):
        """
        信息熵
        :param dataset:
        :return:
        """
        entry_num = len(dataset)
        label = dataset[:, -1]
        label_count = {}
        for l in label:
            if l not in label_count:
                label_count[l] = 1
            else:
                label_count[l] += 1
        result = 0
        for key in label_count:
            prob = label_count[key] / entry_num
            result += -prob * log(prob, 2)
        return result

    def dataset_split(self, dataset, dim):
        """

        :param dataset:
        :param dim: begin from 0
        :return: like{
        "0":dataset0
        "1":dataset1
        }
        """
        # 根据特征划分数据集
        dataset_dict = {}
        dataset_dim = dataset[:, dim]
        for i in range(len(dataset_dim)):
            if dataset_dim[i] not in dataset_dict:
                dataset_dict[dataset_dim[i]] = [dataset[i].tolist()]
            else:
                dataset_dict[dataset_dim[i]].append(dataset[i].tolist())

        return dataset_dict

    def condition_entropy(self, dataset, dim):
        """
        信息增益
        :param dataset:
        :param dim:
        :return:
        """
        dataset_dict = self.dataset_split(dataset, dim)
        entropy = self.entropy(dataset)
        condition_entropy = 0
        for key in dataset_dict:
            set = np.array(dataset_dict[key])
            Di = len(set)
            D = len(dataset)
            entr = self.entropy(set)
            condition_entropy += (Di / D) * entr
        return entropy - condition_entropy

    def best_dim(self, dataset):
        dim_num = len(dataset[0, 0:-1])
        max_dim = {}
        for dim in range(dim_num):
            condition_entropy = self.condition_entropy(dataset, dim)
            max_dim[dim] = condition_entropy
        result = sorted(max_dim.items(), key=lambda x: x[1], reverse=True)

        return result[0][0], result[0][1]

    def max_entry(self, dataset):
        max_dict = {}
        for data in dataset:
            if data not in max_dict:
                max_dict[data] = 1
            else:
                max_dict[data] += 1
        return sorted(max_dict.items(), key=lambda x: x[1], reverse=True)[0][0]

    def train(self, train_data):
        """

        :param train_data: DateFrame
        :return:决策树
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True,
                        label=y_train.iloc[0])
        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature,max_info_gain = self.best_dim(np.array(train_data))
        max_feature_name = features[max_feature]
        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 5,构建Ag子集
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)
        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)
            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        # pprint.pprint(node_tree.tree)
        return node_tree



    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree
    def predict(self, x_test):
        return self._tree.predict(x_test)


id3 = Id3()
tree = id3.fit(data_df)
print(tree.predict(['青年', '否', '否', '一般']))
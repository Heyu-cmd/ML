import pandas as pd
import numpy as np
import sys
from copy import deepcopy


class Node:
    def __init__(self, data, left, right, feature, split, label):
        """

        :param data: 节点所包含的数据
        :param left: 左孩子
        :param right: 右孩子
        :param split: 这个节点的划分点值是多少
        :param feature:这个节点是通过哪个特征划分的
        :param label:类别标签，只有叶子节点有这个属性
        """
        self.data = data
        self.left = left
        self.right = right
        self.split = split
        self.feature = feature
        self.label = label


class Cart:
    def __init__(self, dataframe, type=False):
        """
        :param dataframe: 输入为dataframe格式的数据
        :param type: True-回归树 False-分类树，默认是分类树
        """
        self.feature_names = dataframe.columns[:-1]
        dataset = np.array(dataframe)
        self.train = dataset[:, :]
        self.label = len(self.feature_names)
        self.features = [i for i in range(len(self.feature_names))]
        if not type:
            # 创建分类树
            self.root = self.create_classification_tree(self.train)
        else:
            self.root = self.create_regression_tree(self.train)
        self.roots = []  # 存放子树序列
        self.a = sys.maxsize
        self.minnode = None

    def is_one_class(self, train):
        # 如果数据集中所有实例属于同一类，返回True
        if len(set(train[:, -1])) == 1:
            return True

    def max_num_calss(self, data):
        label_dict = self.feature_num(data, -1)
        max_num_label = max(label_dict.items(), key=lambda x: x[1])[0]
        return max_num_label

    def feature_num(self, data, feature):
        label = data[:, feature]
        label_dict = {}
        for l in label:
            if l not in label_dict:
                label_dict[l] = 1
            else:
                label_dict[l] += 1
        return label_dict

    def cal_gini(self, data):
        # 计算一个数据集的gini系数
        D = data.shape[0]
        label_dict = self.feature_num(data, -1)
        sum_ = 0
        for k, v in label_dict.items():
            sum_ += (v / D) ** 2
        return 1 - sum_

    def feature_gini(self, data, feature):
        """
        计算某个特征的gini系数
        :param data: 数据集
        :param feature: 特征下标
        :return: gini系数，划分数据左，划分数据右，划分值
        """
        min_gini = sys.maxsize
        feature_value = None
        set_left = None
        set_right = None
        raw_data = set(data[:, feature])

        for k in raw_data:
            d_k = data[(data[:, feature] == k), :]  # 特征feature取值是k的集合
            d_not_k = data[(data[:, feature] != k), :]  # 特征feature取值不是k的集合
            D1 = d_k.shape[0]
            D2 = d_not_k.shape[0]
            gini = D1 * self.cal_gini(d_k) + D2 * self.cal_gini(d_not_k)
            if gini < min_gini:
                min_gini = gini
                feature_value = k
                set_left = d_k
                set_right = d_not_k
        return min_gini, feature_value, set_left, set_right

    def create_classification_tree(self, data):
        """
        分类树
        :param train: 训练集，包含特征集和标记
        :return: 分类树
        """
        min_gini = sys.maxsize
        min_split = None
        min_feature = None
        min_left = None
        min_right = None

        if data.shape[0] == 0:
            # 如果数据集中没有数据
            return None
        elif self.is_one_class(data):
            # 如果训练集中数据属于同一类
            node = Node(data=data, left=None, right=None, feature=self.label, split=None, label=data[0, -1])
        elif len(self.features) == 0:
            # 没有可供划分的特征
            node = Node(data=data, left=None, right=None, feature=self.label, split=None,
                        label=self.max_num_calss(data))
        else:
            for i in self.features:
                gini, value, left, right = self.feature_gini(data, i)
                if gini < min_gini:
                    min_gini = gini
                    min_split = value
                    min_feature = i
                    min_left = left
                    min_right = right
            self.features.remove(min_feature)
            left_node = self.create_classification_tree(min_left)
            right_node = self.create_classification_tree(min_right)
            node = Node(data, left_node, right_node, min_feature, min_split, None)

        return node

    def cost_function(self, data):
        # 计算回归树的损失函数
        label = data[:, -1]
        label = label.astype(int)
        avg = np.sum(label) / label.shape[0]
        lost = 0
        for i in range(label.shape(0)):
            # avg为预测值，label[i]为实际值，计算平方误差和
            lost += (label[i] - avg) ** 2
        return lost

    def create_regression_tree(self, data):
        # 生成回归树
        min_cost = sys.maxsize
        min_split = None
        min_feature_index = None
        min_left = None
        min_right = None
        if data.shape[0] == 0:
            # 如果数据集中没有数据
            return None
        elif self.is_one_class(data):
            # 如果训练集中数据属于同一类
            node = Node(data=data, left=None, right=None, feature=self.label, split=None, label=data[0, -1])
        else:
            for i in self.features:
                d = data[np.argsort(data[:, i])]  # 针对第i个特征进行排序，顺序为从小到大
                for j in range(d.shape[0] - 1):
                    left = d[:j]
                    right = d[j:]
                    left_cost = self.cost_function(left)
                    right_cost = self.cost_function(right)
                    if left_cost + right_cost < min_cost:
                        min_split = data[j, i]
                        min_feature_index = i
                        min_left = left
                        min_right = right
            self.features.remove(min_feature_index)
            left_node = self.create_regression_tree(min_left)
            right_node = self.create_regression_tree(min_right)
            node = Node(data, left_node, right_node, min_feature_index, min_split, None)
        return node

    def if_need_prune(self, node):
        # 判断是否需要继续剪枝
        # 如果子节点不是叶子节点，则返回True，继续剪枝
        if node.left.feature != self.label or node.right.feature != self.label:  # 还有内部结点
            return True
        else:
            return False

    def copy_tree(self, node):
        # 创建以t为根节点的子树
        copy_tree = None
        if node == None:
            copy_tree = None
        else:
            left = self.copy_tree(node.left)
            right = self.copy_tree(node.right)
            copy_tree = Node(node.data, left, right, node.feature, node.split, node.label)
        return copy_tree

    def error(self, dataset):
        feature_num = self.feature_num(dataset, -1)
        max_class = self.max_num_calss(dataset)
        # 计算预测误差
        err = 0
        for k,v in feature_num.items():
            if k != max_class:
                # 该数据集中： 错误分类/总实例数  * 该数据集实例数/总数据集实例数
                err += (v/dataset.shape[0])*(dataset.shape[0]/self.train.shape[0])
        return err

    def cal_prune_what(self, node):
        """
        自上而下对各内部结点t计算C(Tt)，｜Tt｜和gt
        :param node:结点
        :return:
        """
        leaf_error = 0
        leaf_num = 0
        # 如果结点的左右孩子都是None，说明到达树底，返回叶子结点对应的数据集的误差，和 leafnum
        if node.left == None and node.right == None:  # 如果该结点是叶子结点
            return self.error(node.data), 1
        else:
            # 如果该结点不是叶子结点且有左孩子
            if node.left != None:
                # 继续向下递归
                left_error, left_num = self.cal_prune_what(node.left)
                # 将返回的误差和叶子结点数加到总误差和总叶子数
                leaf_error += left_error
                leaf_num += left_num
            if node.right != None:
                right_error, right_num = self.cal_prune_what(node.right)
                leaf_error += right_error
                leaf_num += right_num
            # 这是Ct
            node_err = self.error(node)
            gt = (node_err - leaf_error) / (leaf_num - 1)
            # 结点为root结点是不参与计算，因为T0已经加入在rootslist里
            if node != self.root:
                # 选择最小的a和gt
                if gt < self.a:
                    self.a = gt
                    # 更新minnode
                    self.minnode = node
            return leaf_error, leaf_num

    def prune(self):
        # 后剪枝
        k = 0
        root = self.root
        now_tree = self.copy_tree(root)
        self.roots.append(now_tree)
        while self.if_need_prune(root):
            self.a = sys.maxsize
            self.minnode = None
            self.cal_prune_what(root)
            self.minnode.left= None
            self.minnode.right = None
            self.minnode.feature = -1
            self.split = None
            self.value = self.max_num_calss(self.minnode.data)
            now_tree = self.copy_tree(root)
            self.roots.append(now_tree)

    def pre_order(self, node):
        if node != None:
            if node.left == None and node.right == None:  # 如果是叶子结点
                print(str(node.value) + "\n")
            else:
                print(self.feature_name[node.feature])
                self.pre_order(node.left)
                self.pre_order(node.right)

    def level_order(self):
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            if node.left == None and node.right == None:
                print(node.value)
            else:
                print(self.feature_name[node.feature])
                queue.append(node.left)
                queue.append(node.right)

    def predict(self, tree, test):
        while tree.left != None or tree.right != None:
            feature = tree.feature
            split = tree.split
            if test[feature] == split:
                tree = tree.left
            else:
                tree = tree.right
        print("result:", tree.label)


datasets = np.array([['青年', '否', '否', '一般', 0],
                     ['青年', '否', '否', '好', 0],
                     ['青年', '是', '否', '好', 1],
                     ['青年', '是', '是', '一般', 1],
                     ['青年', '否', '否', '一般', 0],
                     ['中年', '否', '否', '一般', 0],
                     ['中年', '否', '否', '好', 0],
                     ['中年', '是', '是', '好', 1],
                     ['中年', '否', '是', '非常好', 1],
                     ['中年', '否', '是', '非常好', 1],
                     ['老年', '否', '是', '非常好', 1],
                     ['老年', '否', '是', '好', 1],
                     ['老年', '是', '否', '好', 1],
                     ['老年', '是', '否', '非常好', 1],
                     ['老年', '否', '否', '一般', 1],
                     ['青年', '否', '否', '一般', 1]])
feature_name = ['年龄', '有工作', '有自己房子', '信用情况', '标签']
df = pd.DataFrame(datasets, columns=feature_name)
cart = Cart(df)
cart.predict(cart.root, np.array(['中年', '否', '是', '非常好']))

import numpy as np
import pandas as pd
from sklearn import datasets
import sys
class Gx:
    def __init__(self,feature,split,left,right):
        #一个分布 这个分布是以哪个feature 作为划分的 划分点 split是 这个分布的系数是coef
         self.feature=feature
         self.split= split
         self.left =left#小于切分点的值为多少
         self.right = right #大于切分点的值为多少
class GDBT:
    def __init__(self,Train,Test,e):
        self.Train=Train #训练集
        self.Test=Test #测试集
        self.feature_num = self.Train.shape[1] - 2  # 求特征个数 因为训练数据的最后一列是标签 所以特征数等于维度数减一
        self.label = self.Train.shape[1] - 2  # 标签所在列
        self.r = self.Train.shape[1] - 1  # 残差所在列
        self.Gx=[] #基分类器
        #self.M = M #迭代次数
        self.e=e # 阈值 在训练数据的损失函数值小于e时停止训练
        self.split = self.cal_split()

    def cal_split(self):
        # 因为数据的特征都是连续值 计算每一个特征的切分点 每两个数据的中点集合作为这个特征的切点集合
        split = {}
        for i in range(self.feature_num):
            split[i] = []  # 将某一个特征的所有切分点放到一个列表中
            d = self.Train[np.argsort(self.Train[:, i])]  # 以这个特征的大小来进行排序
            for j in range(d.shape[0] - 1):
                sp = (d[j][i] + d[j + 1][i]) / 2
                if sp not in split[i]:  # 可能有的切分点是重复的 这一步去重
                    split[i].append(sp)
        return split

    def cost(self, data): #求损失值
        X = data[:, self.r:self.r + 1]
        avg = np.sum(X) / X.shape[0]  # 平均值
        sum = 0
        for i in range(X.shape[0]):
            sum += (X[i][0] - avg) ** 2
        return sum,avg

    def update_r(self,G): #更新残差值
        feature=G.feature
        split=G.split
        left=G.left
        right=G.right
        for i in range(self.Train.shape[0]):
            if self.Train[i][feature]  <= split:
                self.Train[i][self.r]=self.Train[i][self.r]-left
            else:
                self.Train[i][self.r] = self.Train[i][self.r] - right

    def cal_best_split(self):
        min_cost=sys.maxsize
        min_split=0
        min_feature=0
        min_left=0
        min_right=0
        #选择最佳特征，以及最佳切分点
        for i in range(self.feature_num):  # 遍历所有特征取值
             for j in self.split[i]:#遍历每一个切分点
                 left = self.Train[(self.Train[:, i] <= j), :]
                 right = self.Train[(self.Train[:, i] > j), :]
                 cost1,avg1=self.cost(left)
                 cost2,avg2=self.cost(right)
                 # print(cost1)
                 # print(cost2)
                 # print("******")
                 if cost1+cost2 < min_cost:
                     min_cost= cost1+cost2
                     min_feature=i
                     min_split=j
                     min_left=avg1
                     min_right=avg2
        G=Gx(min_feature,min_split,min_left,min_right)
        self.update_r(G)
        self.Gx.append(G)

    def fit(self):
        L=sys.maxsize
        while L>self.e:
            self.cal_best_split()
            print("当前损失值:")
            L=self.cal_cost(self.Train)
            print(L)
            print("***************")

    def print_Gx(self):
        for gx in self.Gx:
            print("特征:")
            print(gx.feature)
            print("切分点")
            print(gx.split)
            print("左值")
            print(gx.left)
            print("右值")
            print(gx.right)
            print("******************************")

    def cal_cost(self,data):
        L=0#预测损失值
        for i in range(data.shape[0]):
            sum=0# 预测值
            for gx in self.Gx:
                feature=gx.feature
                split=gx.split
                left=gx.left
                right=gx.right
                if data[i][feature] <=split:
                    sum+=left
                else:
                    sum+=right
            L+=(data[i][self.label]-sum)**2
        return L

boston=datasets.load_boston()
data1=boston.data
print(data1.shape)
data2=boston.target
data2.resize([data2.shape[0],1])
data3=np.concatenate((data1,data2), axis=1)
data=np.concatenate((data3,data2), axis=1)
print(data.shape)
print(data)
train_size = int(len(data) * 0.7)#划分训练集与测试集
train = np.array(data[:train_size])
# # print(X_train.shape)
test=np.array(data[train_size:])
model=GDBT(train,test,100)
model.fit()


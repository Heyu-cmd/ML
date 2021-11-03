from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['label'] = iris['target']
data = np.array(df.iloc[:, :])
train, test = train_test_split(data, test_size=0.1)


class NaiveBayes:
    def __init__(self, train, r=1):
        self.train = train
        self.r = r

    def prior_probability(self):
        """

        :return: dict[label_name:probability]
        """
        labels = train[:, -1]
        label_dict = {}
        label_set = set(labels)
        for l in labels:
            if l not in label_dict:
                label_dict[l] = 1
            else:
                label_dict[l] += 1
        prior_dict = {}
        for l in label_set:
            prior_dict[l] = (label_dict[l] + self.r) / (len(labels) + self.r * len(label_set))
        return prior_dict

    def condition_probability(self):
        data = self.train[:, 0:-1]
        labelset = set(self.train[:, -1])
        labelList = list(self.train[:, -1])
        dimNum = len(data[0])
        # characterVal存储了每个特征下的所有无重复值
        characterVal = []
        for i in range(dimNum):
            temp = []
            for j in range(len(data)):
                if data[j][i] not in temp:
                    temp.append(data[j][i])
            characterVal.append(temp)
        probability = []
        for dim in range(dimNum):
            tempMemories = {}
            for val in characterVal[dim]:
                for label in labelset:
                    labelCount = 0
                    conCount = 0
                    for i in range(len(labelList)):
                        if labelList[i] == label:
                            labelCount += 1
                            if data[i][dim] == val:
                                conCount += 1
                    tempMemories[str(val) + "|" + str(label)] = (conCount + self.r) / (
                            labelCount + self.r * len(characterVal[dim]))
            probability.append(tempMemories)
        return probability

    def naive_bayes(self, raw_data):
        prior_probability = self.prior_probability()
        condition_probability = self.condition_probability()
        label_list = set(self.train[:, -1])
        res_dict = {}
        for label in label_list:
            result = prior_probability[label]
            for dim in range(len(raw_data)):
                result *= condition_probability[dim][str(raw_data[dim]) + "|" + str(label)]
            res_dict[label] = result
        value = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)
        return value[0][0]


nb = NaiveBayes(train)
score = 0
M = len(test[:, 0])
for i in range(len(test[:, 0])):
    try:
        pre = nb.naive_bayes(test[i, 0:-1])
        print("data:")
        print(test[i])
        print("actual class:"+str(test[i, -1]))
        print("predict: "+str(pre))
        if pre == test[i, -1]:
            score += 1
    except KeyError:
        print("Fail to predict")
        M -= 1
print("\nScore:")
print(score / M)

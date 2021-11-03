from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt

iris = datasets.load_iris()
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df['label'] = iris['target']
data = np.array(df.iloc[:100, [0, 1, -1]])
train, test = train_test_split(data, test_size=0.4)
x0 = [j for i, j in enumerate(train) if j[-1] == 0]
x1 = [j for i, j in enumerate(train) if j[-1] == 1]


class Node:
    def __init__(self, data_, depth=0, lchild=None, rchild=None):
        self.data = data_
        self.depth = depth
        self.lchild = lchild
        self.rchild = rchild


class KdTree:
    def __init__(self):
        self.KdTree = None
        self.n = 0
        self.nearest = None

    def create(self, data_array, depth=0):
        if len(data_array) > 0:
            m, n = np.shape(data_array)
            self.n = n - 1
            axis = depth % n
            mid = int(m / 2)
            datasetcopy = sorted(data_array, key=lambda x: x[axis])
            node = Node(datasetcopy[mid], depth)
            if depth == 0:
                self.KdTree = node
            node.lchild = self.create(datasetcopy[0:mid], depth + 1)
            node.rchild = self.create(datasetcopy[mid + 1:], depth + 1)
            return node
        return None

    def search(self, x, count=1):
        nearest = []
        for i in range(count):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)

        def recurve(node):
            if node is not None:
                axis = node.depth % self.n
                daxis = x[axis] - node.data[axis]
                if daxis < 0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)
                dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x, node.data)))
                for i, d in enumerate(nearest):
                    if d[0] < 0 or d[0] > dist:
                        self.nearest = np.insert(self.nearest, i, [dist, node], axis=0)
                        self.nearest = self.nearest[:-1]
                        break
                n = list(self.nearest[:,0]).count(-1)
                if self.nearest[-n-1,0] > abs(daxis):
                    if daxis >0:
                        recurve(node.rchild)
                    else:
                        recurve(node.lchild)
        recurve(self.KdTree)
        knn = self.nearest[:,1]
        belong = []
        for i in knn:
            belong.append(i.data[-1])
        b = max(set(belong), key=belong.count)

        return self.nearest,b
kdt = KdTree()
kdt.create(train)
score = 0
for t in test:
    nearest, b = kdt.search(t[:-1],)
    if b == t[-1]:
        score += 1
    print("test:")
    print(t, "predict:", b)
    print("nearest:")
    for n in nearest:
        print(n[1].data, "dist:", n[0])
print(score/len(test))

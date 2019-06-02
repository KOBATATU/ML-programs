
import numpy as np
from utils.funcsions.functions import Sigmoid

class LogisticRegression(object):
    def __init__(self, random_seed=None, lr=0.01, iterations=100):
        self.lr = lr
        self.random_seed = random_seed
        self.iterations = iterations
        self.sigmoid = Sigmoid()

    def InitialWeight(self, x):
        # 重みは正規分布で生み出す
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        features = x.shape[1]
        self.w = np.random.randn(features, )  # 1*D (D,)
        self.b = np.random.randn()  # b (1,)

    def fit(self, x, y):
        self.InitialWeight(x)
        for i in range(self.iterations):
            a = self.b + np.dot(x, self.w)  # activation
            y_prediction = self.sigmoid(a)

            self.w -= self.lr * np.dot((y_prediction - y), x)  # 勾配計算
            self.b -= self.lr * np.sum(y_prediction - y)

    def score(self, true, pred):
        pred = self.sigmoid(self.b + np.dot(pred, self.w))
        pred = np.where(pred > 0.5, 1, 0)
        true_rate = (true == pred).sum()
        return true_rate / len(pred)

    def predict(self, x):
        a = self.b + np.dot(x, self.w)  # activation
        return self.sigmoid(a)

import numpy as np

class BaseRegression(object):

    def __init__(self, random_state=None, lr=0.01, iteration=100, gradient_descent=False):
        self.random_state = random_state
        self.lr = lr
        self.iteration = iteration
        self.gradient_descent = gradient_descent

    def InitialWeight(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.W = np.random.randn(n_features, )

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        if self.gradient_descent:
            #勾配計算はうまくいかないので正則化　今後実装する
            self.InitialWeight(X)

            for i in range(self.iteration):
                pred = np.dot(X, self.W)
                self.W = self.W - self.lr * np.dot((pred - y), X)
        else:
            self.W = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T).dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        pred = X.dot(self.W)
        return pred


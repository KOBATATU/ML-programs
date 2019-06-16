import numpy as np


class BayesianRegression(object):
    def __init__(self,sigma_y = 0.1,sigma_phi = 0.9):
        self.sigma_y = sigma_y #yの共分散の重み
        self.sigma_phi = sigma_phi #パラメータの共分散の重み

    def gaussian(self,X):
        s = 0.1
        c = np.arange(0,1+s,s)
        return np.append(1,np.exp(-0.5 * (X-c)**2/s**2))

    def fit(self,X,y):
        gauss = [self.gaussian(x) for x in X]
        self.sigma = np.linalg.inv(self.sigma_phi*np.eye(X.shape[1]) + self.sigma_y*np.dot(gauss.T,gauss))
        self.mu = self.sigma_y*np.dot(self.sigma,np.dot(gauss.T,y))

    def predict(self,X):
        pred = [np.dot(self.mu,self.gaussian(x)) for x in X]
        return pred
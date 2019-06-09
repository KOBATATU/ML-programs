import numpy as np

class GaussianMixtureModel(object):
    def __init__(self, components, iterations=10, seed=None, cov="diag"):
        """
        :param components: クラスの数
        :param iterations: EMアルゴリズムの適用回数
        :param seed: 乱数の指定
        :param cov: 対角行列かフルで考えるか
        """
        self.components = components
        self.iterations = iterations
        self.parameters = []
        self.responsibilities = []
        self.cov = cov
        self.sample_assignments = None
        self.responsibility = None
        self.seed = seed

    def InitialWeight(self, X):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.priors = np.ones(self.components) / self.components
        n_features = X.shape[1]

        for _ in range(self.components):
            params = {}
            params["mean"] = np.random.uniform(X.min(), X.max(), (n_features,))
            if self.cov == "diag":
                params["cov"] = np.eye(n_features) * 10  # diag covariance
            else:
                params["cov"] = (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))  # full covariance

            self.parameters.append(params)

    def gauss(self, X, params):
        n_features = np.shape(X)[1]
        mean = params["mean"]
        cov = params["cov"]
        determinant = np.linalg.det(cov)
        likelihoods = np.zeros(np.shape(X)[0])
        for i, sample in enumerate(X):
            coeff = (1.0 / (pow((2.0 * np.pi), n_features / 2)
                            * np.sqrt(determinant)))
            exponent = np.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(cov)).dot((sample - mean)))
            likelihoods[i] = coeff * exponent

        return likelihoods

    def likelihood(self, X):
        n_samples = X.shape[0]
        likelihoods = np.zeros((n_samples, self.components))

        for i in range(self.components):
            likelihoods[:, i] = self.gauss(X, self.parameters[i])

        return likelihoods

    # E-step
    def expectations(self, X):
        weighted_likelihoods = self.likelihood(X) * self.priors

        # 負担率の計算
        sum_likelihoods = np.expand_dims(
            np.sum(weighted_likelihoods, axis=1), axis=1)

        self.responsibility = weighted_likelihoods / sum_likelihoods
        # Assign samples to cluster that has largest probability

    # M-step
    def maximization(self, X):
        for i in range(self.components):
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            mean = np.sum(resp * X, axis=0) / np.sum(resp)
            cov = (X - mean).T.dot((X - mean) * resp) / resp.sum()
            self.parameters[i]["mean"] = mean
            self.parameters[i]["cov"] = cov

        self.priors = self.responsibility.sum(axis=0) / len(X)

    def fit(self, X):
        self.InitialWeight(X)

        for _ in range(self.iterations):
            weight = self.priors
            self.expectations(X)
            self.maximization(X)
            if np.allclose(weight, self.priors):
                break

    def classifier(self, X):
        n_samples = X.shape[0]
        likelihoods = np.zeros((n_samples, self.components))
        for i in range(self.components):
            likelihoods[:, i] = self.gauss(X, self.parameters[i])
        posterior = self.priors * likelihoods
        return np.argmax(posterior, axis=1)


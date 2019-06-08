import numpy as np

def train_test_split(X, y, train_size=0.8, random_seed=None, shuffle=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    train_size = len(X) * train_size
    if shuffle is not None:
        perm = np.random.permutation(len(X))
    else:
        perm = [i for i in range(len(X))]

    X_train,X_test = X[perm[:int(train_size)], :],X[perm[int(train_size):], :]
    y_train,y_test = y[perm[:int(train_size)]],y[perm[int(train_size):]]

    return X_train, X_test, y_train, y_test


def shuffle(X, y, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    perm = [i for i in range(len(X))]
    np.random.shuffle(perm)
    return X[perm], y[perm]



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import plot_decision_regions
from numpy.random import seed

df = pd.read_csv('/Users/matt/Desktop/ML/Perceptron/iris.data', header=None)
y = df.iloc[0:100, 4].values  # iloc.values prints only values without description of columns. Here we are choosing the type of flower.
y = np.where(y == 'Iris-setosa', -1, 1)  # returns elements chosen from x or y depending on condition. Here, if y is setosa.
X = df.iloc[0:100, [0, 2]].values


class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, random_state=None, shuffled=True):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_initialized = False
        self.shuffled = shuffled

    def fit(self, X, y):
        self.initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffled:
                X, y = self.shuffle(X, y)
            cost = []

            for x_i, target in zip(X, y):
                cost.append(self.update_weights(x_i, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):  # fits without weight initialization
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for x_i, target in zip(X,y):
                self.update_weights(x_i, target)
        else:
            self.update_weights(X, y)
        return self

    def shuffle(self, X, y):  # shuffles the data.
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)

        self.w_initialized = True

    def update_weights(self, x_i, target):
        output = self.activation(self.net_input(x_i))
        error = (target - output)
        self.w_[1:] += self.eta * x_i.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/ X[:, 1].std()  # for each characteristics.

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel("Length of sepal [standardised]")
plt.ylabel("Length of petal [standardised]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker="o")
plt.xlabel("Iterations")
plt.ylabel("Sum of squares of errors")
plt.show()
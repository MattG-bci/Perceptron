import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/matt/Desktop/ML/Perceptron/iris.data', header=None)
y = df.iloc[0:100, 4].values  # iloc.values prints only values without description of columns. Here we are choosing the type of flower.
y = np.where(y == 'Iris-setosa', -1, 1)  # returns elements chosen from x or y depending on condition. Here, if y is setosa.
X = df.iloc[0:100, [0, 2]].values


class Perceptron(object):
    def __init__(self, eta=0.01, iter_n=50, random_state=1):
        self.eta = eta
        self.iter_n = iter_n
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # Draw random samples from a normal (Gaussian) distribution
        self.errors_ = []

        for i in range(self.iter_n):
            errors = 0
            for x_i, target in zip(X, y):  # zips iterables and returns a tuple.
                update = self.eta * (target - self.predict(x_i))
                self.w_[1:] += update * x_i
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


ppn = Perceptron(eta=0.1, iter_n=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='x')
plt.xlabel("Iterations")
plt.ylabel("Number of updates")
plt.show()

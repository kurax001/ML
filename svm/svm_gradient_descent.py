import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# SVM classifier for donut data using project gradient descent for optimization.
class SVM:
    def __init__(self, C, Kernel):
        self.C = C
        self.Kernel = Kernel

    # decision function
    def classifier(self, X):
        return (self.lagr_multipliers * self.Y).dot(self.Kernel(self.X, X)) + self.b

    # dual svm optimization function
    def optimization_function(self, X, Y):
        return np.sum(self.lagr_multipliers) - 0.5 * np.sum(
            self.Kernel(self.X, X) * np.outer(self.Y, Y) * np.outer(self.lagr_multipliers, self.lagr_multipliers))

    # training svm
    def fit(self, X, Y, learning_rate=1e-4, iterations=300):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.lagr_multipliers = np.random.randn(self.N)
        self.b = 0

        loss_per_iteration = []
        for _ in range(iterations):
            loss = self.optimization_function(X, Y)
            loss_per_iteration.append(loss)
            grads_lagr_multiplier = np.ones(self.N) - self.lagr_multipliers.dot((np.outer(Y, Y) * self.Kernel(X, X)))

            self.lagr_multipliers += learning_rate * grads_lagr_multiplier

            # clip the values of lagrangian multipliers to be between 0 and C.
            self.lagr_multipliers[self.lagr_multipliers < 0] = 0
            self.lagr_multipliers[self.lagr_multipliers > self.C] = self.C

        # calculate bias using the data points with lagrangian multipliers between 0 and C (support vectors).
        idx = np.where((self.lagr_multipliers > 0) & (self.lagr_multipliers < self.C))[0]
        b = Y[idx] - (self.lagr_multipliers * Y).dot(self.Kernel(X, X[idx]))
        self.b = np.mean(b)

        plt.plot(loss_per_iteration)
        plt.title("loss per iteration")
        plt.show()

    def predict(self, X):
        return np.sign(self.classifier(X))

    def score(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)


# generates donut shaped data
def get_donut():
    N = 200
    rd1 = 5
    rd2 = 10

    theta1 = 2 * np.pi * np.random.random(N)
    r1 = np.random.randn(N) + rd1

    x1 = r1 * np.cos(theta1)
    x2 = r1 * np.sin(theta1)
    X = np.stack((x1, x2), axis=1)

    theta2 = 2 * np.pi * np.random.random(N)
    r2 = np.random.randn(N) + rd2
    x1 = r2 * np.cos(theta2)
    x2 = r2 * np.sin(theta2)
    xt = np.stack((x1, x2), axis=1)

    X = np.concatenate((X, xt), axis=0)
    Y = np.array([-1] * N + [1] * N)
    return X, Y


# rbf kernel
def rbf(X1, X2, gamma=0):
    a = np.sum(np.square(X1), axis=1)[:, np.newaxis]
    b = np.sum(np.square(X2), axis=1)
    mul = np.dot(X1, X2.T)
    dists = a + b - 2 * mul
    return np.exp(-gamma * dists)


# plot separating hyperplane
def plot_separating_hyperplane(svmc, x, y):
    xd = np.linspace(x[:, 0].min(), x[:, 0].max())
    yd = np.linspace(x[:, 1].min(), x[:, 1].max())
    xx, yy = np.meshgrid(xd, yd)
    zz = svmc.classifier(np.vstack((xx.ravel(), yy.ravel())).T)
    zz = zz.reshape(xx.shape)

    plt.contour(xd, yd, zz, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    plt.scatter(x[:, 0], x[:, 1], c=y, lw=0, alpha=0.4, cmap='seismic')

    idx = svmc.lagr_multipliers > 0
    plt.scatter(svmc.X[:, 0][idx], svmc.X[:, 1][idx], c=svmc.Y[idx], s=100, linewidth=1,
                facecolors='none', edgecolors='k')
    plt.show()


if __name__ == '__main__':
    x, y = get_donut()
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # normalize the data.
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    kernel = lambda a, b: rbf(a, b, gamma=5)

    svmc = SVM(C=3, Kernel=kernel)
    svmc.fit(x_train, y_train, 1e-5, 200)
    print("train score: ", svmc.score(x_train, y_train))
    print("test score: ", svmc.score(x_test, y_test))
    plot_separating_hyperplane(svmc, x_train, y_train)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearSVM:
    def __init__(self, C):
        self.C = C

    # svm decision function
    def classifier(self, x):
        return x.dot(self.w) + self.b

    # Soft margin optimization function
    def optimization_function(self, functional_margin):
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - functional_margin).sum()

    # training svm
    def fit(self, x, y, learning_rate=1e-4, iterations=300):
        N, D = x.shape
        self.w = np.random.randn(D)
        self.b = 0
        self.N = N

        loss_per_iteration = []
        for _ in range(iterations):
            functional_margin = y * self.classifier(x)
            loss = self.optimization_function(functional_margin)
            loss_per_iteration.append(loss)

            # SVM uses hinge loss ( max(0, 1-y*f(x)). The learning problem is equivalent to the unconstrained
            # optimization problem over w and b with the soft margin classifier. calculate gradients of the
            # unconstrained optimization function w.r.t w and b. The training points that are have functional margin
            # > 1 are correctly classified. Those training points do not contribute to the loss. The training points
            # with functional margin < 1 violates the margin ( and are misclassified if the functional margin is <
            # 0), contribute to the loss.
            #  w_gradient = w - c * yx (if margin < 1)
            #  w_gradient = w (otherwise).

            indices = np.where(functional_margin < 1)[0]
            w_gradient = self.w - self.C * y[indices].dot(x[indices])
            b_gradient = -self.C * y[indices].sum()
            self.w = self.w - learning_rate * w_gradient
            self.b = self.b - learning_rate * b_gradient

        support_indices = np.where((y * self.classifier(x)) <= 1)[0]
        self.support_vectors = x[support_indices]
        plt.plot(loss_per_iteration)
        plt.title("SVM loss per iteration")
        plt.show()

    def predict(self, x):
        return np.sign(self.classifier(x))

    def score(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions == y)


# Generate data - two gaussian clouds that are linearly separable.
def get_data():
    N = 800
    # gaussian cloud centered at (3,3)
    d1 = np.random.randn(N, 2) + np.array([2, 2])
    # gaussian cloud centered at (-2,-2)
    d2 = np.random.randn(N, 2) + np.array([-2, -2])
    x = np.concatenate([d1, d2])
    y = np.array([-1] * N + [1] * N)
    return x, y


def plot_separating_hyperplane(svmc, x, y):
    xd = np.linspace(x[:, 0].min(), x[:, 0].max())
    yd = np.linspace(x[:, 1].min(), x[:, 1].max())
    xx, yy = np.meshgrid(xd, yd)
    zz = svmc.classifier(np.vstack((xx.ravel(), yy.ravel())).T)
    zz = zz.reshape(xx.shape)

    plt.contour(xd, yd, zz, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    plt.scatter(x[:, 0], x[:, 1], c=y, lw=0, alpha=0.4, cmap='seismic')

    idx = svmc.support_vectors
    plt.scatter(idx[:, 0], idx[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


if __name__ == '__main__':
    x, y = get_data()
    print(x.shape, y.shape)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # normalize the data.
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)

    svmc = LinearSVM(C=3)
    svmc.fit(x_train, y_train)
    print("train score: ", svmc.score(x_train, y_train))
    print("test score: ", svmc.score(x_test, y_test))
    plot_separating_hyperplane(svmc, x_train, y_train)

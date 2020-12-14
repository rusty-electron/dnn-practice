import numpy as np

class Perceptron:
    def __init__ (self, N, alpha=0.1, debug=False):
        # the division is done in order to scale the weight matrix, that leads to faster convergence
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
        self.debug = debug

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # employ bias trick and add a column of 1s
        X = np.c_[X, np.ones(X.shape[0])]
        if self.debug:
            print(f"[debug] shape of np.ones array is {X.shape[0]}")

        for epochs in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                # take dot product
                p = self.step(np.dot(x, self.W))

                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias = True):

        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones(X.shape[0])]

        if self.debug:
            print(f"[debug] shape of W is {np.array(self.W).shape}")
            print(f"[debug] shape of X is {np.array(X).shape}")
        return self.step(np.dot(X, self.W))

# TODO: document the matrix dimensions and how they are multiplied



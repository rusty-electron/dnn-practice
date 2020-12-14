import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1, debug=False):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        self.debug = debug

        for i in np.arange(0, len(layers) - 2):
            # creates a 3x3 matrix
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # creates a 3x1 matrix
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

        # W is now a numpy array made of sub-arrays 3x3 and 3x1 matrix
        # len of W is 2

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # X is 4x2 matrix of possible 0 & 1 combination
        X = np.c_[X, np.ones(X.shape[0])]
        # the matrix now is 4x3 matrix with an extra column of 1s

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                # x is a 1x3 matrix and target is a 1x1 matrix e.g. [[0]]
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)] # A = [ array([[0, 1, 1]]) ]

        # FEEDFORWARD
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer]) # 1x3 matmul with 3x3 mat to yield a 1x3 mat

            out = self.sigmoid(net)
            A.append(out) # predictions appended to A

            # in second pass, A[1] is a 1x3 mat and W[1] is a 3x1, hence we get a single output
        # end of FF
        if self.debug:
            print(f"Value of A:{A}")

            # BACKPROPAGATION
        error = A[-1] - y # A[-1] is 1x1 mat

        D = [error * self.sigmoid_deriv(A[-1])]

        # application of chain-rule
        for layer in np.arange(len(A) - 2, 0, -1):
           # len of A is 3

           delta = D[-1].dot(self.W[layer].T)
           delta = delta * self.sigmoid_deriv(A[layer])
           D.append(delta)

           # pass 1: D[-1] is 1x1 mat, dotted with W[1].T = 3x1.T = 1x3 so delta is 1x3 mat
           # A[1] is 1x3 mat

           # pass 2: D[-1] is 1x3 mat, dotted with W[0].T = 3x3.T = 3x3 so delta is 1x3 mat
           # A[0] is 1x3 mat

        D = D[::-1]
        if self.debug:
            print(f"Value of D:{D}")

        # weight update phase
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss

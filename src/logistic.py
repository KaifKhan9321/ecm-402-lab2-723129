import numpy as np

class MyLogisticRegression:
    """Binary logistic regression via batch gradient descent, L2-regularized, NumPy only."""

    def __init__(self, alpha=0.0, lr=0.1, epochs=2000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.loss_history_ = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _loss(self, X, y):
        n = len(y)
        yhat = self._sigmoid(X @ self.w + self.b)
        eps = 1e-12
        bce = -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))
        reg = (self.alpha / (2 * n)) * np.sum(self.w ** 2)
        return bce + reg

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        history = np.empty(self.epochs)
        for epoch in range(self.epochs):
            yhat = self._sigmoid(X @ self.w + self.b)
            error = yhat - y
            grad_w = (X.T @ error) / n + (self.alpha / n) * self.w
            grad_b = np.mean(error)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            history[epoch] = self._loss(X, y)
        self.loss_history_ = history
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
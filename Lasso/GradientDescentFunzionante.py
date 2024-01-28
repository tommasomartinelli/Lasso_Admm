import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LassoRegression:
    def __init__(self, step_size, max_iterations, l1_penalty, tolerance):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.l1_penalty = l1_penalty
        self.tolerance = tolerance
        self.m = None
        self.n = None
        self.W = None
        self.X = None
        self.Y = None
        self.iterations = None
        self.J = None

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.X = X
        self.Y = Y
        self.iterations = 0
        self.J = np.zeros(self.max_iterations)

        for i in range(self.max_iterations):
            Y_predict = self.predict(self.X)

            soft_term = self.soft_threshold(self.W, self.l1_penalty)
            dW = (-2 * self.X.T @ (self.Y - Y_predict) + soft_term) / self.m
            new_W = self.W - self.step_size * dW

            if np.mean(np.abs(new_W - self.W)) < self.tolerance:
                break

            self.J[i] = np.mean(np.abs(new_W - self.W))
            self.W = new_W
            self.iterations = i

    def predict(self, X):
        return X @ self.W

    def soft_threshold(self, w, th):
        return np.maximum(0, w - th) - np.maximum(0, -w - th)
    
dataset = pd.read_csv('Lasso/dataset.csv')
dataset.iloc[:, [0, 2, 3]] = (dataset.iloc[:, [0, 2, 3]] - dataset.iloc[:, [0, 2, 3]].min()) / (dataset.iloc[:, [0, 2, 3]].max() - dataset.iloc[:, [0, 2, 3]].min())

# Split dei dati (80% train, 20% test)
cv = np.random.rand(len(dataset)) < 0.8
train = dataset[cv]
test = dataset[~cv]

X_train = train.iloc[:, :9].values
Y_train = train.iloc[:, 9].values
X_test = test.iloc[:, :9].values
Y_test = test.iloc[:, 9].values

iterations = 50000
step_size = 0.01
l1_penalty = 1
tolerance = 1e-4

lasso = LassoRegression(step_size, iterations, l1_penalty, tolerance)
lasso.fit(X_train, Y_train)

print("Numero di iterazioni:", lasso.iterations)
print("Pesi appresi:", lasso.W)

Y_predicted = lasso.predict(X_test)

r_squared = np.corrcoef(Y_test, Y_predicted)**2
print("Coefficienti di determinazione (R^2):", r_squared)

plt.scatter(Y_test, Y_predicted)
plt.plot(Y_test, Y_test, color='red', linewidth=2)
plt.xlabel('Valori effettivi')
plt.ylabel('Valori predetti')
plt.title('Predizioni vs. Valori effettivi')
plt.show()

# Plot della loss durante le iterazioni
plt.plot(lasso.J)
plt.title("Convergenza Gradient Descent")
plt.xlabel("Iterazioni")
plt.ylabel("Loss")
plt.show()
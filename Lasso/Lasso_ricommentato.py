import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LassoReg:
    def __init__(self, step_size, max_iterations, l1_penalty, tolerance):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.l1_penalty = l1_penalty
        self.tolerance = tolerance
        self.m = None
        self.n = None
        self.A = None
        self.X = None
        self.Y = None
        self.J = []  # Initialize J as an empty list
        self.iterations = None

    def fit(self, X, Y, algo, agents=None):
        self.m, self.n = X.shape
        self.A = np.zeros(self.n)
        self.X = X
        self.Y = Y

        if algo == "gd":
            self.gradient_descent()
        elif algo == "admm":
            self.admm_fit()
        elif algo == "dist":
            self.distributed_admm(agents)
    def admm_fit(self):
        rho = self.step_size
        z = np.zeros(self.n)
        u = np.zeros(self.n)
        I = np.eye(self.n)

        abs_tol = self.tolerance
        rel_tol = abs_tol * 100

        for i in range(1, self.max_iterations + 1):
            last_z = z

            # Aggiornamento di X
            self.X = np.linalg.solve(self.A @ self.A.T + rho * I, self.A @ self.Y + rho * (z - u))

            # Aggiornamento di A
            self.A = np.linalg.solve(self.X.T @ self.X + rho * I, self.X.T @ self.Y + rho * (z - u))

            z = self.soft_threshold(self.X + u, self.l1_penalty / rho)
            u = u + self.X - z

            r_norm = np.linalg.norm(self.X - z)  # primary residual
            s_norm = np.linalg.norm(-rho * (z - last_z))  # dual residual
            tol_prim = np.sqrt(self.n) * abs_tol + rel_tol * max(np.linalg.norm(self.X), np.linalg.norm(-z))
            tol_dual = np.sqrt(self.n) * abs_tol + rel_tol * np.linalg.norm(rho * u)

            self.iterations = i
            self.J.append((r_norm, s_norm, tol_prim, tol_dual))

            if r_norm < tol_prim and s_norm < tol_dual:
                break

        self.X = self.X.reshape(1, -1)
        self.A = self.A.reshape(1, -1)
    
    def gradient_descent(self):
        for i in range(1, self.max_iterations + 1):
            Y_predict = self.predict(self.X).flatten()

            soft_term = self.soft_threshold(self.A, self.l1_penalty)
            dA = (-2 * self.X.T @ (self.Y - Y_predict) + soft_term) / self.m
            new_A = self.A - self.step_size * dA

            if np.mean(np.abs(new_A - self.A)) < self.tolerance:
                break

            self.J.append((np.mean(np.abs(new_A - self.A)),))
            self.A = new_A
            self.iterations = i

    def distributed_admm(self, agents):
        rho = self.step_size
        z = np.zeros(self.n)
        I = np.eye(self.n)

        abs_tol = self.tolerance
        rel_tol = abs_tol * 100
        converged = False

        r, c = self.X.shape
        rows_per_agent = r // agents
        total_rows_used = rows_per_agent * agents

        print("Original X shape:", self.X.shape)
        print("Total rows used:", total_rows_used)

        splitted_X = self.X[:total_rows_used, :].reshape((rows_per_agent, agents, c))
        splitted_Y = np.reshape(self.Y[:total_rows_used], (rows_per_agent, agents))
        self.A = np.zeros((agents, c))
        u = np.zeros((agents, c))

        for i in range(1, self.max_iterations + 1):
            last_z = z
            for j in range(agents):
                self.A[j, :] = np.linalg.solve(splitted_X[:, j, :].T @ splitted_X[:, j, :] + rho * I,
                                               splitted_X[:, j, :].T @ splitted_Y[:, j] + rho * (z - u[j, :]))
            z = self.soft_threshold(np.mean(self.A, axis=0) + np.mean(u, axis=0), self.l1_penalty / rho)
            for j in range(agents):
                u[j, :] = u[j, :] + (self.A[j, :] - z)

            r_norm = np.linalg.norm(np.mean(self.A, axis=0) - z)  # primary residual
            s_norm = np.linalg.norm(-rho * (z - last_z))  # dual residual
            tol_prim = np.sqrt(self.n) * abs_tol + rel_tol * max(np.linalg.norm(np.mean(self.A, axis=0)),
                                                                 np.linalg.norm(-z))
            tol_dual = np.sqrt(self.n) * abs_tol + rel_tol * np.linalg.norm(rho * np.mean(u, axis=0))

            self.J.append((r_norm, s_norm, tol_prim, tol_dual))

            if r_norm < tol_prim and s_norm < tol_dual:
                converged = True
                break

            self.iterations = i

        if not converged:
            print("ADMM did not converge within the specified number of iterations.")

        self.A = np.mean(self.A, axis=0).reshape(1, -1)

    def predict(self, X):
        return X @ self.A.T.flatten()

    def loss_function(self, Y, Y_predict, A):
        return 0.5 * np.sum((Y - Y_predict) ** 2) + self.l1_penalty * np.linalg.norm(W, 1)

    def soft_threshold(self, a, th):
        return np.maximum(0, a - th) - np.maximum(0, -a - th)

    def mean_squared_error(self, Y_true, Y_predicted):
        return np.mean((Y_true - Y_predicted) ** 2)

def plot_predict(label, Y_test, Y_predicted):
    plt.figure()
    plt.title(label)
    plt.scatter(Y_test, Y_predicted)
    plt.plot(Y_test, Y_test, '--')
    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')
    plt.show()

def plot_loss(lasso, label):
    if label == "Loss GD":
        plt.figure()
        plt.title(label)
        plt.plot(lasso.J)
        plt.plot(np.full_like(lasso.J, lasso.tolerance), "--")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()
    else:
        J_arr = np.array(lasso.J)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title(label)
        plt.plot(J_arr[:, 0])
        plt.plot(J_arr[:, 2], "--")
        plt.xlabel('Iterations')
        plt.ylabel('Primary residual')

        plt.subplot(2, 1, 2)
        plt.plot(J_arr[:, 1])
        plt.plot(J_arr[:, 3], "--")
        plt.xlabel('Iterations')
        plt.ylabel('Dual residual')

        plt.show()

# Load data
dataset = pd.read_csv('Lasso/dataset.csv')
# Normalize data between [0,1]
dataset.iloc[:, [0, 2, 3]] = (dataset.iloc[:, [0, 2, 3]] - dataset.iloc[:, [0, 2, 3]].min()) / (
        dataset.iloc[:, [0, 2, 3]].max() - dataset.iloc[:, [0, 2, 3]].min())

# Data split (train: 80%, test: 20%) -> randomized!
cv = np.random.rand(len(dataset)) < 0.8
train = dataset[cv]
test = dataset[~cv]

X_train = train.iloc[:, :9].values
Y_train = train.iloc[:, 9].values
X_test = test.iloc[:, :9].values
Y_test = test.iloc[:, 9].values

# Parameters
iterations = 50000
step_size = 0.01
l1_penalty = 1
tolerance = 1e-4

# Soft-thresholding Lasso with Gradient Descent
print("GD")
lasso = LassoReg(step_size, iterations, l1_penalty, tolerance)
lasso.fit(X_train, Y_train, "gd")
print(lasso.iterations)
Y_predicted = lasso.predict(X_test)
print(np.corrcoef(Y_test, Y_predicted)[0, 1] ** 2)  # R2

# Plot prediction
plot_predict("Lasso GD", Y_test, Y_predicted)
plot_loss(lasso, "Loss GD")

# Soft-thresholding Lasso with ADMM
print("ADMM")
lasso_admm = LassoReg(step_size, iterations, l1_penalty, tolerance)
lasso_admm.fit(X_train, Y_train, "admm")
print(lasso_admm.iterations)
Y_predicted_admm = lasso_admm.predict(X_test)

# Make Y_test and Y_predicted_admm one-dimensional arrays
Y_test = Y_test.flatten()
Y_predicted_admm = Y_predicted_admm.flatten()

# Check dimensions before calling np.corrcoef
if Y_test.shape == Y_predicted_admm.shape:
    r2_admm = np.corrcoef(Y_test, Y_predicted_admm)[0, 1] ** 2
    print(r2_admm)
else:
    print("Dimensions of Y_test and Y_predicted_admm do not match.")

# Plot prediction
plot_predict("Lasso ADMM", Y_test, Y_predicted_admm)
plot_loss(lasso_admm, "Convergence ADMM")

# Soft-thresholding Lasso with Distributed ADMM
print("Distributed ADMM")
agents = 10
lasso_dist = LassoReg(step_size, iterations, l1_penalty, tolerance)
lasso_dist.fit(X_train, Y_train, "dist", agents)
print(lasso_dist.iterations)
Y_predicted_dist = lasso_dist.predict(X_test)

# Check dimensions before calling np.corrcoef
print("Shape Y_test:", Y_test.shape)
print("Shape Y_predicted_dist:", Y_predicted_dist.shape)

if Y_test.shape == Y_predicted_dist.shape:
    r2_dist = np.corrcoef(Y_test, Y_predicted_dist)[0, 1] ** 2
    print(r2_dist)
else:
    print("Dimensions of Y_test and Y_predicted_dist do not match.")
    # Add additional dimension information if needed
    print("Dimension of Y_test:", Y_test.shape)
    print("Dimension of Y_predicted_dist:", Y_predicted_dist.shape)

# Plot prediction
plot_predict("Lasso Distributed-ADMM", Y_test, Y_predicted_dist)
plot_loss(lasso_dist, "Convergence Distributed-ADMM")

# Calculate Mean Squared Error (MSE)
print("MSE GD:", lasso.mean_squared_error(Y_test, Y_predicted))

# For ADMM model
print("MSE ADMM:", lasso_admm.mean_squared_error(Y_test, Y_predicted_admm))

# For Distributed ADMM model
print("MSE Distributed ADMM:", lasso_dist.mean_squared_error(Y_test, Y_predicted_dist))

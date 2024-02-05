import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import time

class LassoReg:
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
        self.J = []  # Initialize J as an empty list
        self.iterations = None

    def fit(self, X, Y, algo, agents=None):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.X = X
        self.Y = Y

        if algo == "gd":
            self.gradient_descent()
        elif algo == "admm":
            self.admm_fit()
        else:
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

            self.W = np.linalg.solve(self.X.T @ self.X + rho * I, self.X.T @ self.Y + rho * (z - u))
            z = self.soft_threshold(self.W + u, self.l1_penalty / rho)
            u = u + self.W - z

            r_norm = np.linalg.norm(self.W - z)  # primary residual
            s_norm = np.linalg.norm(-rho * (z - last_z))  # dual residual
            tol_prim = np.sqrt(self.n) * abs_tol + rel_tol * max(np.linalg.norm(self.W), np.linalg.norm(-z))
            tol_dual = np.sqrt(self.n) * abs_tol + rel_tol * np.linalg.norm(rho * u)

            self.iterations = i
            self.J.append((r_norm, s_norm, tol_prim, tol_dual))

            if r_norm < tol_prim and s_norm < tol_dual:
                break

        self.W = self.W.reshape(1, -1)

    def gradient_descent(self):
        for i in range(1, self.max_iterations + 1):
            Y_predict = self.predict(self.X).flatten()

            soft_term = self.soft_threshold(self.W, self.l1_penalty)
            dW = (-2 * self.X.T @ (self.Y - Y_predict) + soft_term) / self.m
            new_W = self.W - self.step_size * dW

            if np.mean(np.abs(new_W - self.W)) < self.tolerance:
                break

            self.J.append((np.mean(np.abs(new_W - self.W)),))
            self.W = new_W
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
        self.W = np.zeros((agents, c))
        u = np.zeros((agents, c))

        for i in range(1, self.max_iterations + 1):
            last_z = z
            for j in range(agents):
                self.W[j, :] = np.linalg.solve(splitted_X[:, j, :].T @ splitted_X[:, j, :] + rho * I,
                                            splitted_X[:, j, :].T @ splitted_Y[:, j] + rho * (z - u[j, :]))
            
            # Fusion center aggregates information from agents
            global_W = np.mean(self.W, axis=0)
            global_u = np.mean(u, axis=0)

            # Fusion center updates z based on aggregated information
            z = self.soft_threshold(global_W + global_u, self.l1_penalty / rho)

            # Distribute updated information to agents
            for j in range(agents):
                u[j, :] = u[j, :] + (self.W[j, :] - z)
                
            global_u = np.mean(u, axis=0)    

            r_norm = np.linalg.norm(global_W - z)  # primary residual
            s_norm = np.linalg.norm(-rho * (z - last_z))  # dual residual
            tol_prim = np.sqrt(self.n) * abs_tol + rel_tol * max(np.linalg.norm(global_W), np.linalg.norm(-z))
            tol_dual = np.sqrt(self.n) * abs_tol + rel_tol * np.linalg.norm(rho * global_u)

            self.iterations = i
            self.J.append((r_norm, s_norm, tol_prim, tol_dual))

            if r_norm < tol_prim and s_norm < tol_dual:
                converged = True
                break

        self.W = global_W.reshape(1, -1)

    def distributed_admm_feature(self, agents):
        rho = self.step_size
        z = np.zeros(self.n)
        I = np.eye(self.n)

        abs_tol = self.tolerance
        rel_tol = abs_tol * 100
        converged = False

        r, c = self.X.shape
        cols_per_agent = c // agents

        splitted_X = self.X[:, :agents * cols_per_agent].reshape((r, agents, cols_per_agent))
        splitted_Y = np.reshape(self.Y, (r, agents))

        for i in range(1, self.max_iterations + 1):
            last_z = z
            for j in range(agents):
                # Local Lasso optimization at each processor
                X_local = splitted_X[:, j, :]
                Y_local = splitted_Y[:, j]
                A_local = X_local.T @ X_local
                b_local = X_local.T @ Y_local.flatten()

                # Update local variable
                x_local = np.linalg.solve(A_local + rho * np.eye(cols_per_agent), b_local + rho * (z - self.u[j, :]))
                self.W[:, j * cols_per_agent:(j + 1) * cols_per_agent] = x_local.reshape(1, -1)

            # Fusion center aggregates information from agents
            global_W = np.mean(self.W, axis=0)
            global_u = np.mean(self.u, axis=0)

            # Fusion center updates z based on aggregated information
            z = self.soft_threshold(global_W + global_u, self.l1_penalty / rho)

            # Distribute updated information to agents
            for j in range(agents):
                self.u[j, :] = self.u[j, :] + (self.W[:, j * cols_per_agent:(j + 1) * cols_per_agent] - z)

            r_norm = np.linalg.norm(global_W - z)  # primary residual
            s_norm = np.linalg.norm(-rho * (z - last_z))  # dual residual
            tol_prim = np.sqrt(self.n) * abs_tol + rel_tol * max(np.linalg.norm(global_W), np.linalg.norm(-z))
            tol_dual = np.sqrt(self.n) * abs_tol + rel_tol * np.linalg.norm(rho * global_u)

            self.iterations = i
            self.J.append((r_norm, s_norm, tol_prim, tol_dual))

            if r_norm < tol_prim and s_norm < tol_dual:
                converged = True
                break

        self.W = global_W.reshape(1, -1)

    '''def distributed_admm_multithreaded(self, agents):
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
        self.W = np.zeros((agents, c))
        u = np.zeros((agents, c))

        def solve_subproblem(j):
            self.W[j, :] = np.linalg.solve(splitted_X[:, j, :].T @ splitted_X[:, j, :] + rho * I,
                                            splitted_X[:, j, :].T @ splitted_Y[:, j] + rho * (z - u[j, :]))
            u[j, :] = u[j, :] + (self.W[j, :] - z)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Esegui gli aggiornamenti di ogni agente in modo concorrente
            executor.map(solve_subproblem, range(agents))

        # Fusion center aggrega le informazioni dagli agenti, aggiorna z, distribuisce u
        global_W = np.mean(self.W, axis=0)
        global_u = np.mean(u, axis=0)
        z = self.soft_threshold(global_W + global_u, self.l1_penalty / rho)

        for j in range(agents):
            u[j, :] = u[j, :] + (self.W[j, :] - z)

        r_norm = np.linalg.norm(global_W - z)  # primary residual
        s_norm = np.linalg.norm(-rho * (z - z_last))  # dual residual
        tol_prim = np.sqrt(self.n) * abs_tol + rel_tol * max(np.linalg.norm(global_W), np.linalg.norm(-z))
        tol_dual = np.sqrt(self.n) * abs_tol + rel_tol * np.linalg.norm(rho * global_u)

        self.iterations = i
        self.J.append((r_norm, s_norm, tol_prim, tol_dual))

        if r_norm < tol_prim and s_norm < tol_dual:
            converged = True

        self.W = global_W.reshape(1, -1) '''

    def predict(self, X):
        return X @ self.W.T.flatten()

    def loss_function(self, Y, Y_predict, W):
        return 0.5 * np.sum((Y - Y_predict) ** 2) + self.l1_penalty * np.linalg.norm(W, 1)

    def soft_threshold(self, w, th):
        return np.maximum(0, w - th) - np.maximum(0, -w - th)

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
start_time = time.time()
print("GD")
lasso = LassoReg(step_size, iterations, l1_penalty, tolerance)
lasso.fit(X_train, Y_train, "gd")
print(lasso.iterations)
Y_predicted = lasso.predict(X_test)
print(np.corrcoef(Y_test, Y_predicted)[0, 1] ** 2)  # R2
end_time = time.time()
print("Time Gradient Descent:", end_time - start_time)
# Plot prediction
plot_predict("Lasso GD", Y_test, Y_predicted)
plot_loss(lasso, "Loss GD")



# Soft-thresholding Lasso with ADMM
start_time = time.time()
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
end_time = time.time()
print("Time ADMM:", end_time - start_time)
# Plot prediction
plot_predict("Lasso ADMM", Y_test, Y_predicted_admm)
plot_loss(lasso_admm, "Convergence ADMM")



# Soft-thresholding Lasso with Distributed ADMM
start_time = time.time()
print("Distributed ADMM")
agents = 6
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
end_time = time.time()
print("Time Distributed ADMM:", end_time - start_time)
# Plot prediction
plot_predict("Lasso Distributed-ADMM", Y_test, Y_predicted_dist)
plot_loss(lasso_dist, "Convergence Distributed-ADMM")


# Soft-thresholding Lasso with Distributed ADMM (Multithreaded)
start_time = time.time()
print("Distributed ADMM (Multithreaded)")
agents = 6
lasso_dist_mt = LassoReg(step_size, iterations, l1_penalty, tolerance)
lasso_dist_mt.fit(X_train, Y_train, "dist", agents)
print(lasso_dist_mt.iterations)
Y_predicted_dist_mt = lasso_dist_mt.predict(X_test)

# Check dimensions before calling np.corrcoef
print("Shape Y_test:", Y_test.shape)
print("Shape Y_predicted_dist_mt:", Y_predicted_dist_mt.shape)

if Y_test.shape == Y_predicted_dist_mt.shape:
    r2_dist_mt = np.corrcoef(Y_test, Y_predicted_dist_mt)[0, 1] ** 2
    print(r2_dist_mt)
else:
    print("Dimensions of Y_test and Y_predicted_dist_mt do not match.")
    # Add additional dimension information if needed
    print("Dimension of Y_test:", Y_test.shape)
    print("Dimension of Y_predicted_dist_mt:", Y_predicted_dist_mt.shape)
end_time = time.time()
print("Time Distributed ADMM (Multithreaded):", end_time - start_time)
# Plot prediction
plot_predict("Lasso Distributed-ADMM (Multithreaded)", Y_test, Y_predicted_dist_mt)
plot_loss(lasso_dist_mt, "Convergence Distributed-ADMM (Multithreaded)")

#distributed admm splitted by features
start_time = time.time()
print("Distributed ADMM (Feature)")
agents = 6
lasso_dist_feature = LassoReg(step_size, iterations, l1_penalty, tolerance)
lasso_dist_feature.fit(X_train, Y_train, "dist", agents)
print(lasso_dist_feature.iterations)
Y_predicted_dist_feature = lasso_dist_feature.predict(X_test)

print("Shape Y_test:", Y_test.shape)
print("Shape Y_predicted_dist_feature:", Y_predicted_dist_feature.shape)

if Y_test.shape == Y_predicted_dist_feature.shape:
    r2_dist_feature = np.corrcoef(Y_test, Y_predicted_dist_feature)[0, 1] ** 2
    print(r2_dist_feature)
else:
    print("Dimensions of Y_test and Y_predicted_dist_feature do not match.")
    print("Dimension of Y_test:", Y_test.shape)
    print("Dimension of Y_predicted_dist_feature:", Y_predicted_dist_feature.shape)
end_time = time.time()
print("Time Distributed ADMM (Feature):", end_time - start_time)

plot_predict("Lasso Distributed-ADMM (Feature)", Y_test, Y_predicted_dist_feature)
plot_loss(lasso_dist_feature, "Convergence Distributed-ADMM (Feature)")


# Calculate Mean Squared Error (MSE)
print("MSE GD:", lasso.mean_squared_error(Y_test, Y_predicted))

# For ADMM model
print("MSE ADMM:", lasso_admm.mean_squared_error(Y_test, Y_predicted_admm))

# For Distributed ADMM model
print("MSE Distributed ADMM:", lasso_dist.mean_squared_error(Y_test, Y_predicted_dist))

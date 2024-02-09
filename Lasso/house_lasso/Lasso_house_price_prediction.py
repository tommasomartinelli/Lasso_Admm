import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

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
        elif algo == "distributed_admm":
            self.distributed_admm(agents)
        else:
            self.admm_splitted_by_features(X, Y, self.l1_penalty, self.step_size, self.max_iterations, self.tolerance, agents)


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
                self.W[j, :] = np.linalg.solve(splitted_X[:, j, :].T @ splitted_X[:, j, :] + (rho / 2) * I, splitted_X[:, j, :].T @ splitted_Y[:, j] + (rho / 2) * (z - u[j, :]))
            
            # Fusion center aggregates information from agents
            global_W = np.mean(self.W, axis=0)
            global_u = np.mean(u, axis=0)

            # Fusion center updates z based on aggregated information
            z = self.soft_threshold(global_W + global_u, self.l1_penalty / (rho * agents))

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

    def admm_splitted_by_features(self, X, Y, lambda_val, rho, max_iterations, tolerance, agents):
        num_instances, num_features = X.shape
        features_per_agent = num_features // agents
        
        # Split features among agents
        splitted_X = np.array_split(X, agents, axis=1)
        
        # Initialize variables
        W = np.zeros((agents, num_features))
        z_bar = np.mean(W, axis=0)  # Fusion center variable
        u_bar = np.zeros_like(z_bar)  # Fusion center variable
        
        for iteration in range(1, max_iterations + 1):
            W_updated = np.zeros_like(W)
            X_W_bar = np.mean(np.matmul(X, W.T), axis=0)  # Fusion center variable
            
            # Update W distributedly
            for agent in range(agents):
                def objective_function(w_i, lambda_val, rho, X_i, w_i_k, z_k, X_W_bar_k, u_k):
                    # Formula per l'aggiornamento di W
                    l1_penalty_term = lambda_val * np.linalg.norm(w_i, 1)
                    admm_residual_term = (rho / 2) * np.linalg.norm(X_i @ w_i - (X_i @ w_i_k + z_k - X_W_bar_k - u_k))**2
                    return l1_penalty_term + admm_residual_term

                result = minimize(objective_function, np.zeros_like(W[agent]), 
                                args=(lambda_val, rho, splitted_X[agent], W[agent], z_bar, X_W_bar, u_bar),
                                method='L-BFGS-B')
                W_updated[agent] = result.x
            
            W = W_updated
            
            # Update z_bar and u_bar at the fusion center
            z_bar_updated = (1 / (num_instances + rho / 2)) * (Y + (rho / 2) * num_instances * X_W_bar + rho * num_instances * u_bar)
            u_bar_updated = u_bar + (X_W_bar - z_bar_updated)
            
            # Check convergence
            r_norm = np.linalg.norm(np.mean(W, axis=0) - z_bar_updated)
            s_norm = np.linalg.norm(rho * (z_bar_updated - z_bar))
            tol_prim = np.sqrt(num_features) * tolerance + tolerance * max(np.linalg.norm(np.mean(W, axis=0)), np.linalg.norm(z_bar_updated))
            tol_dual = np.sqrt(num_features) * tolerance + tolerance * np.linalg.norm(rho * u_bar_updated)
            
            if r_norm < tol_prim and s_norm < tol_dual:
                break
            
            z_bar = z_bar_updated
            u_bar = u_bar_updated
        
        return np.mean(W, axis=0), z_bar, u_bar
    '''   versione splitted by features inizata ora from scipy.optimize import minimize

    def objective_function(w_i, lambda_val, rho, X_i, w_i_k, z_k, X_W_bar_k, u_k):
        term1 = lambda_val * np.linalg.norm(w_i, 1)
        term2 = (rho / 2) * np.linalg.norm(X_i @ w_i - (X_i @ w_i_k + z_k - X_W_bar_k - u_k), 2) ** 2
        return term1 + term2

    # Dati del problema
    lambda_val = 1.0
    rho = 0.1
    X_i = np.random.rand(10, 5)  # Sostituisci con la tua matrice delle features
    w_i_k = np.random.rand(5)  # Sostituisci con il vettore dei pesi corrispondente all'iterazione k
    z_k = np.random.rand(5)  # Sostituisci con il vettore medio delle variabili ausiliarie a k
    X_W_bar_k = np.random.rand(10)  # Sostituisci con il vettore medio delle predizioni Xw a k
    u_k = np.random.rand(5)  # Sostituisci con il vettore medio delle variabili di aggiornamento duale a k

    # Ottimizzazione per trovare l'argmin
    result = minimize(objective_function, np.zeros_like(w_i_k), args=(lambda_val, rho, X_i, w_i_k, z_k, X_W_bar_k, u_k), method='L-BFGS-B')
    w_i_updated = result.x

    print("Pesetti aggiornati:", w_i_updated)


    def update_z(y, rho, N, X_W_bar_kplus1, u_bar_kplus1):
    term1 = y
    term2 = (rho / 2) * N * X_W_bar_kplus1
    term3 = (rho / 2) * N * u_bar_kplus1
    denominator = N + rho / 2
    
    z_bar_kplus1 = (term1 + term2 + term3) / denominator
    
    return z_bar_kplus1

    def update_u(u_bar_k, X_W_bar_kplus1, z_bar_kplus1):
    u_bar_kplus1 = u_bar_k + (X_W_bar_kplus1 - z_bar_kplus1)
    return u_bar_kplus1
    '''

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
dataset = pd.read_csv('Lasso/house_lasso/cleaned_data.csv')

X = dataset.drop(columns=['price'])  
Y = dataset['price']

#Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Parameters
iterations = 200000
step_size = 0.01
l1_penalty = 1
tolerance = 1e-4
agents = 3

# Soft-thresholding Lasso with ADMM Splitted by Features
start_time = time.time()
print("ADMM Splitted by Features")
lasso_admm_splitted = LassoReg(step_size, iterations, l1_penalty, tolerance)
lasso_admm_splitted.fit(X_train, Y_train, "admm_splitted", agents)
print(lasso_admm_splitted.iterations)
Y_predicted_admm_splitted = lasso_admm_splitted.predict(X_test)

# Check dimensions before calling np.corrcoef
if Y_test.shape == Y_predicted_admm_splitted.shape:
    r2_admm_splitted = np.corrcoef(Y_test, Y_predicted_admm_splitted)[0, 1] ** 2
    print(r2_admm_splitted)
else:
    print("Dimensions of Y_test and Y_predicted_admm_splitted do not match.")
    # Add additional dimension information if needed
    print("Dimension of Y_test:", Y_test.shape)
    print("Dimension of Y_predicted_admm_splitted:", Y_predicted_admm_splitted.shape)
end_time = time.time()
print("Time ADMM Splitted by Features:", end_time - start_time)
# Plot prediction
plot_predict("Lasso ADMM Splitted by Features", Y_test, Y_predicted_admm_splitted)
plot_loss(lasso_admm_splitted, "Convergence ADMM Splitted by Features")

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
agents = 5
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


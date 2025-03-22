import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Logistic Regression part
def logistic_loss(w, X, y):

    N = X.shape[0]
    z = X.dot(w)
    p = sigmoid(z)
    eps = 1e-8  
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return loss

def logistic_grad(w, X, y):

    N = X.shape[0]
    z = X.dot(w)
    p = sigmoid(z)
    grad = X.T.dot(p - y) / N
    return grad


#  Two-Layer Neural Network 
def unpack_theta(theta, input_dim, hidden_dim):

    W1_end = input_dim * hidden_dim
    W1 = theta[:W1_end].reshape(input_dim, hidden_dim)
    b1 = theta[W1_end:W1_end + hidden_dim]
    W2 = theta[W1_end + hidden_dim:W1_end + hidden_dim + hidden_dim].reshape(hidden_dim, 1)
    b2 = theta[-1]
    return W1, b1, W2, b2

def nn_forward(theta, X, input_dim, hidden_dim):

    W1, b1, W2, b2 = unpack_theta(theta, input_dim, hidden_dim)
    Z1 = X.dot(W1) + b1       # (N, hidden_dim)
    A1 = np.tanh(Z1)          # activation (tanh)
    Z2 = A1.dot(W2) + b2      # (N, 1)
    A2 = sigmoid(Z2)          # output probability
    return A2, A1, Z1

def nn_loss(theta, X, y, input_dim, hidden_dim):

    N = X.shape[0]
    A2, _, _ = nn_forward(theta, X, input_dim, hidden_dim)
    A2 = A2.flatten()
    eps = 1e-8
    loss = -np.mean(y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps))
    return loss

def nn_grad(theta, X, y, input_dim, hidden_dim):

    N = X.shape[0]
    W1, b1, W2, b2 = unpack_theta(theta, input_dim, hidden_dim)
    A2, A1, Z1 = nn_forward(theta, X, input_dim, hidden_dim)
    A2 = A2.flatten()  # shape (N,)
    y = y.flatten()
    # using bckpropagation !!1
    dZ2 = (A2 - y).reshape(N, 1)          # (N, 1)
    dW2 = A1.T.dot(dZ2) / N               # (hidden_dim, 1)
    db2 = np.sum(dZ2, axis=0) / N          # (1,)
    dA1 = dZ2.dot(W2.T)                   # (N, hidden_dim)
    dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)      # derivative of tanh, (N, hidden_dim)
    dW1 = X.T.dot(dZ1) / N                # (input_dim, hidden_dim)
    db1 = np.sum(dZ1, axis=0) / N          # (hidden_dim,)
    # Flatten all gradients into a single vector
    grad = np.concatenate([dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()])
    return grad


def run_gd(model_type, X, y, lr=0.1, n_iters=100, hidden_dim=5):

    trajectory = []
    grads = []
    losses = []
    
    if model_type == 'logistic':
        # Augment X with bias (last column = 1)
        X_lr = np.hstack([X, np.ones((X.shape[0], 1))])
        param_dim = X_lr.shape[1]
        w = np.random.randn(param_dim) * 0.1
        for i in range(n_iters):
            loss = logistic_loss(w, X_lr, y)
            grad = logistic_grad(w, X_lr, y)
            trajectory.append(w.copy())
            grads.append(grad.copy())
            losses.append(loss)
            w -= lr * grad
        extra_params = {'X': X_lr, 'loss_func': logistic_loss, 'y': y}
        return np.array(trajectory), np.array(grads), losses, extra_params

    elif model_type == 'nn':
        input_dim = X.shape[1]
        # Total parametes: W1 (input_dim*hidden_dim) + b1 (hidden_dim) + W2 (hidden_dim) + b2 (1)
        param_dim = input_dim * hidden_dim + hidden_dim + hidden_dim + 1
        theta = np.random.randn(param_dim) * 0.1
        for i in range(n_iters):
            loss = nn_loss(theta, X, y, input_dim, hidden_dim)
            grad = nn_grad(theta, X, y, input_dim, hidden_dim)
            trajectory.append(theta.copy())
            grads.append(grad.copy())
            losses.append(loss)
            theta -= lr * grad
        extra_params = {'X': X, 'loss_func': nn_loss, 'y': y, 'input_dim': input_dim, 'hidden_dim': hidden_dim}
        return np.array(trajectory), np.array(grads), losses, extra_params

    else:
        raise ValueError("Unknown model type")



def compute(pca, loss_func, extra_params, proj_trajectory, grid_size=30):

    x_min, x_max = proj_trajectory[:, 0].min(), proj_trajectory[:, 0].max()
    y_min, y_max = proj_trajectory[:, 1].min(), proj_trajectory[:, 1].max()
    margin_x = (x_max - x_min) * 0.2
    margin_y = (y_max - y_min) * 0.2
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    x_lin = np.linspace(x_min, x_max, grid_size)
    y_lin = np.linspace(y_min, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    Z_grid = np.zeros_like(X_grid)

    for i in range(grid_size):
        for j in range(grid_size):
            point = np.array([X_grid[i, j], Y_grid[i, j]])
            full_params = pca.inverse_transform(point)
            # For the NN, loss_func requires extra arguments.
            if 'input_dim' in extra_params:
                loss_val = loss_func(full_params, extra_params['X'], extra_params['y'],
                                     extra_params['input_dim'], extra_params['hidden_dim'])
            else:
                loss_val = loss_func(full_params, extra_params['X'], extra_params['y'])
            Z_grid[i, j] = loss_val
    return X_grid, Y_grid, Z_grid

# def animate_trajectory(trajectory, grads, losses, extra_params, model_type, pca, proj_trajectory, grid_size=30):

#     loss_func = extra_params['loss_func']
#     X_grid, Y_grid, Z_grid = compute(pca, loss_func, extra_params, proj_trajectory, grid_size=grid_size)
    
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     n_steps = len(trajectory)
#     for i in range(n_steps):
#         ax.cla()  # clear the axes
#         # Plot the error surface
#         ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.6)
#         # Current parameter (projected)
#         curr_proj = proj_trajectory[i]
#         curr_loss = losses[i]
#         # Project the full gradient into PCA space:
#         proj_grad = grads[i].dot(pca.components_.T)  # shape (2,)
        
#         # Plot the current point and gradient arrow.
#         ax.scatter(curr_proj[0], curr_proj[1], curr_loss, color='r', s=50)
#         scale = 0.1  # scale factor for arrow length
#         ax.quiver(curr_proj[0], curr_proj[1], curr_loss,
#                   scale * proj_grad[0], scale * proj_grad[1], 0,
#                   color='black', arrow_length_ratio=0.3)
        
#         ax.set_xlabel('PC1')
#         ax.set_ylabel('PC2')
#         ax.set_zlabel('Loss')
#         ax.set_title(f"{model_type} - Iteration {i+1}/{n_steps} - Loss: {curr_loss:.4f}")
#         plt.pause(0.2)
#     plt.show()

def animate_trajectory(trajectory, grads, losses, extra_params, model_type, 
                            pca, proj_trajectory, tsne_proj, grid_size=30):

    # New animation: 
    #   - Left: PCA-based visualization.
    #   - Right: TSNE-based visualization: the trajectory (with loss as z) and
    #            an arrow connecting the previous and current TSNE points.

    loss_func = extra_params['loss_func']
    # Compute PCA error surface grid 
    X_grid, Y_grid, Z_grid = compute(pca, loss_func, extra_params, proj_trajectory, grid_size=grid_size)
    
    n_steps = len(trajectory)
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection='3d')  # PCA subplot
    ax2 = fig.add_subplot(122, projection='3d')  # TSNE subplot
    
    losses_arr = np.array(losses)
    
    for i in range(n_steps):
        # PCA Plot (Left)
        ax1.cla()
        ax1.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.6)
        curr_proj = proj_trajectory[i]
        curr_loss = losses[i]
        proj_grad = grads[i].dot(pca.components_.T)
        ax1.scatter(curr_proj[0], curr_proj[1], curr_loss, color='r', s=50)
        scale = 0.1
        ax1.quiver(curr_proj[0], curr_proj[1], curr_loss,
                   scale * proj_grad[0], scale * proj_grad[1], 0,
                   color='black', arrow_length_ratio=0.3)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('Loss')
        ax1.set_title(f"{model_type} PCA - Iteration {i+1}/{n_steps}\nLoss: {curr_loss:.4f}")
        
        # TSNE Plot (Right)
        ax2.cla()
        # For TSNE, we use the pre-computed tsne_proj for x and y, and the loss for z.
        tsne_x = tsne_proj[:i+1, 0]
        tsne_y = tsne_proj[:i+1, 1]
        tsne_z = losses_arr[:i+1]
        ax2.plot(tsne_x, tsne_y, tsne_z, color='blue', marker='o', linestyle='-')
        # Highlight the current point.
        ax2.scatter(tsne_proj[i, 0], tsne_proj[i, 1], losses[i], color='r', s=50)
        if i > 0:
            dx = tsne_proj[i, 0] - tsne_proj[i-1, 0]
            dy = tsne_proj[i, 1] - tsne_proj[i-1, 1]
            dz = losses[i] - losses[i-1]
            ax2.quiver(tsne_proj[i-1, 0], tsne_proj[i-1, 1], losses[i-1],
                       dx, dy, dz, color='black', arrow_length_ratio=0.3)
        ax2.set_xlabel('TSNE1')
        ax2.set_ylabel('TSNE2')
        ax2.set_zlabel('Loss')
        ax2.set_title(f"{model_type} TSNE - Iteration {i+1}/{n_steps}\nLoss: {curr_loss:.4f}")
        
        plt.pause(0.2)
    plt.show()


def main():
    np.random.seed(42)
    # Generate a synthetic binary classification dataset.
    N = 200
    d = 5  # input feature dimension
    X = np.random.randn(N, d)
    # Generate labels with a random hyperplane (for logistic regression)
    true_w = np.random.randn(d)
    logits = X.dot(true_w)
    probs = sigmoid(logits)
    y = (probs > 0.5).astype(float)

    n_iters = 50
    lr = 0.5

    # --- Part 1: Logistic Regression (Convex) ---
    print("GD - Logistic Regression...")
    traj_log, grads_log, losses_log, extra_log = run_gd('logistic', X, y, lr=lr, n_iters=n_iters)
    # Fit PCA on the full trajectory (each parameter vector is high-dimensional)
    pca_log = PCA(n_components=2)
    proj_traj_log = pca_log.fit_transform(traj_log)
    # Also compute TSNE projection on the trajectory (TSNE is computed offline)
    tsne_model_log = TSNE(n_components=2, random_state=42)
    tsne_proj_log = tsne_model_log.fit_transform(traj_log)
    
    # Animate both visualizations simultaneously (side by side)
    animate_trajectory(traj_log, grads_log, losses_log, extra_log,
                            'Logistic Regression', pca_log, proj_traj_log, tsne_proj_log, grid_size=30)

    # --- Part 2: Two-Layer Neural Network (Nonconvex) ---
    print("GD - Two-Layer Neural Network")
    hidden_dim = 5
    traj_nn, grads_nn, losses_nn, extra_nn = run_gd('nn', X, y, lr=lr, n_iters=n_iters, hidden_dim=hidden_dim)
    pca_nn = PCA(n_components=2)
    proj_traj_nn = pca_nn.fit_transform(traj_nn)
    tsne_model_nn = TSNE(n_components=2, random_state=42)
    tsne_proj_nn = tsne_model_nn.fit_transform(traj_nn)
    
    animate_trajectory(traj_nn, grads_nn, losses_nn, extra_nn,
                            'Two-Layer Neural Network', pca_nn, proj_traj_nn, tsne_proj_nn, grid_size=30)

if __name__ == '__main__':
    main()

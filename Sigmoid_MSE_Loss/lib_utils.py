import numpy as np


def sigmoid_2d(X, W, b):
    return 1.0 / (1.0 + np.exp(-(np.dot(X, W) + b)))


def gradient(X, W, b, Y_pred, Y_true, eta=0.2):
    error = (Y_pred - Y_true) * Y_pred * (1 - Y_pred) * 1.0
    # print('error.shape before: ', error.shape)
    error = error.reshape(-1, 1)
    # print('error.shape after: ', error.shape)
    grad_w = np.sum(X * error, axis=0)
    grad_b = np.sum(error)
    # print('W.shape: ', W.shape)
    # print('grad_w.shape: ', grad_w.shape)
    W -= eta * grad_w
    b -= eta * grad_b
    return W, b


def compute_loss(Y, Y_exp):
    return sum((Y - Y_exp) ** 2.0)


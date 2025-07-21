import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class FNN:
    '''
    This uses numpy vectors for calculation
    '''
    def __init__(self):
        print('Numpy vector calculation')
        self.num_weights, self.num_biases = 6, 3
        (self.w1, self.w2, self.w3,
         self.w4, self.w5, self.w6
         ) = np.random.randn(self.num_weights)
        self.b1, self.b2, self.b3 = [0.0] * 3

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def pre_activation(X, b, W):
        return np.dot(X, W) + b

    @staticmethod
    def activation(z):
        return FNN.sigmoid(z)

    @staticmethod
    def forward_pass(w1, w2, w3, w4, w5, w6, b1, b2, b3, X):
        a1 = FNN.pre_activation(X, b1, np.array([w1, w2]))
        h1 = FNN.activation(a1)

        a2 = FNN.pre_activation(X, b2, np.array([w3, w4]))
        h2 = FNN.activation(a2)

        a3 = FNN.pre_activation(np.column_stack((h1, h2)), b3, np.array([w5, w6]))
        h3 = FNN.activation(a3)

        return h1, h2, h3

    @staticmethod
    def dh_da(a):
        return a * (1.0 - a)

    @staticmethod
    def grad(w1, w2, w3, w4, w5, w6, b1, b2, b3, Y, X):
        h1, h2, h3 = FNN.forward_pass(
            w1, w2, w3, w4, w5, w6,
            b1, b2, b3, X
        )
        # print(h3.shape, h2.shape, h1.shape)
        diff_y = h3 - Y

        dh3_da3 = FNN.dh_da(h3)
        dh2_da2 = FNN.dh_da(h2)
        dh1_da1 = FNN.dh_da(h1)

        coeff_a1 = diff_y * dh3_da3 * w5 * dh1_da1
        dL_dw1 = coeff_a1 * X[:, 0]
        dL_dw2 = coeff_a1 * X[:, 1]
        dL_db1 = coeff_a1

        coeff_a2 = diff_y * dh3_da3 * w6 * dh2_da2
        dL_dw3 = coeff_a2 * X[:, 0]
        dL_dw4 = coeff_a2 * X[:, 1]
        dL_db2 = coeff_a2

        coeff_a3 = diff_y * dh3_da3
        dL_dw5 = coeff_a3 * h1
        dL_dw6 = coeff_a3 * h2
        dL_db3 = coeff_a3

        return (dL_dw1, dL_dw2, dL_db1,
                dL_dw3, dL_dw4, dL_db2,
                dL_dw5, dL_dw6, dL_db3
                )

    def fit(self, X, Y, learning_rate=0.5, epochs=1000, display_loss=True, initialise=True):
        losses = []
        if initialise:
            (self.w1, self.w2, self.w3,
             self.w4, self.w5, self.w6
             ) = np.random.randn(self.num_weights)
            self.b1, self.b2, self.b3 = [0.0] * 3

        print('Initial Weights and biases')
        print(self.w1, self.w2, self.w3, self.w4, self.w5, self.w6)
        m = X.shape[0]
        for i in range(epochs):
            if display_loss:
                Y_preds = self.predict(X)
                loss = mean_squared_error(Y, Y_preds)
                losses.append(loss)
            # for x, y in zip(X, Y):
            if True:
                (dL_dw1, dL_dw2, dL_db1,
                 dL_dw3, dL_dw4, dL_db2,
                 dL_dw5, dL_dw6, dL_db3
                 ) = FNN.grad(self.w1, self.w2, self.w3, self.w4, self.w5, self.w6,
                              self.b1, self.b2, self.b3, Y, X)

                self.w1 -= (np.sum(dL_dw1) / (m * 1.0)) * learning_rate
                self.w2 -= (np.sum(dL_dw2) / (m * 1.0)) * learning_rate
                self.w3 -= (np.sum(dL_dw3) / (m * 1.0)) * learning_rate
                self.w4 -= (np.sum(dL_dw4) / (m * 1.0)) * learning_rate
                self.w5 -= (np.sum(dL_dw5) / (m * 1.0)) * learning_rate
                self.w6 -= (np.sum(dL_dw6) / (m * 1.0)) * learning_rate
                self.b1 -= (np.sum(dL_db1) / (m * 1.0)) * learning_rate
                self.b2 -= (np.sum(dL_db2) / (m * 1.0)) * learning_rate
                self.b3 -= (np.sum(dL_db3) / (m * 1.0)) * learning_rate
                # print('gradient w and b')
                # print(np.sum(dL_dw1), np.sum(dL_dw2), np.sum(dL_dw3), np.sum(dL_dw4),
                #       np.sum(dL_dw5), np.sum(dL_dw6), np.sum(dL_db1), np.sum(dL_db2),
                #       np.sum(dL_db3),
                #       )
                # print(f'\ni: {i}, Weights and biases')
                # print(self.w1, self.w2, self.w3, self.w4, self.w5, self.w6)
                #
                # if i == 1:
                #     exit(0)

        if display_loss:
            FNN.display_loss(losses)

    def predict(self, X):
        _, _, y_preds = FNN.forward_pass(
            self.w1, self.w2, self.w3, self.w4,
            self.w5, self.w6, self.b1, self.b2, self.b3, X)
        return y_preds

    @staticmethod
    def display_loss(loss):
        plt.plot(loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()




import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class FNN:
    def __init__(self):
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

        a3 = FNN.pre_activation(np.array([h1, h2]), b3, np.array([w5, w6]))
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
        x1, x2 = X

        diff_y = h3 - Y
        dh3_da3 = FNN.dh_da(h3)
        dh2_da2 = FNN.dh_da(h2)
        dh1_da1 = FNN.dh_da(h1)

        coeff_a1 = diff_y * dh3_da3 * w5 * dh1_da1
        dL_dw1 = coeff_a1 * x1
        dL_dw2 = coeff_a1 * x2
        dL_db1 = coeff_a1

        coeff_a2 = diff_y * dh3_da3 * w6 * dh2_da2
        dL_dw3 = coeff_a2 * x1
        dL_dw4 = coeff_a2 * x2
        dL_db2 = coeff_a2

        coeff_a3 = diff_y * dh3_da3
        dL_dw5 = coeff_a3 * h1
        dL_dw6 = coeff_a3 * h2
        dL_db3 = coeff_a3

        return (dL_dw1, dL_dw2, dL_db1,
                dL_dw3, dL_dw4, dL_db2,
                dL_dw5, dL_dw6, dL_db3
                )

    def fit(self, X, Y, learning_rate=0.05, epochs=1000, display_loss=True, initialise=True):
        losses = []
        m = X.shape[0]
        print('X.shape: ', X.shape)
        if initialise:
            (self.w1, self.w2, self.w3,
             self.w4, self.w5, self.w6
             ) = (np.random.randn(), np.random.randn(), np.random.randn(),
                  np.random.randn(), np.random.randn(), np.random.randn())
        for i in range(epochs):
            if display_loss:
                Y_preds = self.predict(X)
                loss = mean_squared_error(Y, Y_preds)
                if i == 0:
                    print('loss: ', loss)
                losses.append(loss)
            (dw1, dw2, dw3, dw4,
             dw5, dw6, db1, db2, db3) = [0.0] * (self.num_biases+self.num_weights)
            j=0
            for x, y in zip(X, Y):
                (dL_dw1, dL_dw2, dL_db1,
                 dL_dw3, dL_dw4, dL_db2,
                 dL_dw5, dL_dw6, dL_db3
                 ) = FNN.grad(self.w1, self.w2, self.w3, self.w4, self.w5, self.w6,
                              self.b1, self.b2, self.b3, y, x)
                dw1 += dL_dw1
                dw2 += dL_dw2
                dw3 += dL_dw3
                dw4 += dL_dw4
                dw5 += dL_dw5
                dw6 += dL_dw6
                db1 += dL_db1
                db2 += dL_db2
                db3 += dL_db3

                # if j == 1:
                #     exit(0)
                j += 1
            self.w1 -= (dw1 / (m * 1.0)) * learning_rate
            self.w2 -= (dw2 / (m * 1.0)) * learning_rate
            self.w3 -= (dw3 / (m * 1.0)) * learning_rate
            self.w4 -= (dw4 / (m * 1.0)) * learning_rate
            self.w5 -= (dw5 / (m * 1.0)) * learning_rate
            self.w6 -= (dw6 / (m * 1.0)) * learning_rate
            self.b1 -= (db1 / (m * 1.0)) * learning_rate
            self.b2 -= (db2 / (m * 1.0)) * learning_rate
            self.b3 -= (db3 / (m * 1.0)) * learning_rate
            # print(f'\ni={i}: w1, w2, w3, w4, w5, w6, b1, b2, b3')
            # print(dw1, dw2, dw3, dw4, dw5, dw6, db1, db2, db3)
            # if i == 1:
            #     exit(0)
        if display_loss:
            FNN.display_loss(np.array(losses))

    def predict(self, X):
        y_preds = []
        for x in X:
            _, _, y_pred = FNN.forward_pass(
                self.w1, self.w2, self.w3, self.w4,
                self.w5, self.w6, self.b1, self.b2, self.b3, x)
            y_preds.append(y_pred)
        return np.array(y_preds).ravel()

    @staticmethod
    def display_loss(loss):
        plt.plot(loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

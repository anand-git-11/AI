import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def sigmoid_2d(X, W, b):
    return 1.0 / (1.0 + np.exp(-(np.dot(X, W) + b)))


def gradient(X, W, b, Y_pred, Y_true, eta=0.2):
    error = Y_pred - Y_true
    b -= eta * np.sum(error)
    error = error.reshape(-1, 1)
    grad_w = np.sum(X * error, axis=0)
    W -= eta * grad_w
    return W, b


def compute_loss(Y_pred, Y_true):
    # Loss = (1-y)*log(y_pred) + y * log(y_pred)
    return sum(
        (1 - Y_true) * np.log(1 - Y_pred, 2)
        + Y_true * np.log(Y_true, 2)
    )


path = '/Users/user1/AI/mobile_cleaned-1551253091700.csv'
rating_col_name = 'Rating'
threshold = 4.2

scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
data = pd.read_csv(path)
data['Class'] = (data['Rating'] > threshold) * 1
data['Class'].value_counts()

X = data.drop(rating_col_name, axis=1)
X = data.drop('Class', axis=1)
Y = data[rating_col_name].values
Y_binarised = data['Class'].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=1, stratify=Y_binarised
)
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

Y_scaled_train = minmax_scaler.fit_transform(Y_train.reshape(-1, 1))
Y_scaled_test = minmax_scaler.transform(Y_test.reshape(-1, 1))

threshold_scaled = float(minmax_scaler.transform([[4.2]])[0][0])
Y_binarised_train = ((Y_scaled_train >= threshold_scaled) * 1).ravel()
Y_binarised_test = ((Y_scaled_test >= threshold_scaled) * 1).ravel()



losses = {}
epochs = 10
for i in range(1):
    W = np.random.rand(1, X_scaled_train.shape[1])[0]
    b = np.random.random()
    prev_loss = 9 ** 10 * 0.1
    ls = []
    for j in range(epochs):
        Y_train_pred = sigmoid_2d(X_scaled_train, W, b)
        # print('Y_pred.shape: ', Y_pred.shape)
        Y_scaled_train = Y_scaled_train.ravel()
        # print('Y_scaled_train.shape: ', Y_scaled_train)
        # print('Y_scaled_train.shape: ', Y_scaled_train.shape)
        W, b = gradient(
            X_scaled_train,
            W,
            b,
            Y_train_pred,
            Y_scaled_train,
            eta=0.02
        )
        # print('New W.shape: ', W.shape)
        # print(f'W new, W: {W.ravel()}, b: {b}')
        curr_loss = compute_loss(Y_scaled_train, Y_train_pred)
        # print('curr_loss: ', curr_loss, 'prev_loss: ', prev_loss)
        if curr_loss > prev_loss:
            print('breaking')
            break
        prev_loss = curr_loss
        # print('\n\n')
        # break
        ls.append(prev_loss)

    print(f'loss {i}:', prev_loss)
    # plt.plot(ls, '*')
    losses[prev_loss] = (W, b)

min_loss = min(losses.keys())
W, b = losses[min_loss]
print('*'*30)
# print('losses:', losses.keys())
print(f'min loss on training data: {min_loss} for {(W, b)}')
# plt.show()

Y_test_pred = sigmoid_2d(X_scaled_test, W, b)
print('*'*30)
print(Y_test_pred)
print('*'*30)
curr_loss = compute_loss(Y_scaled_test.ravel(), Y_test_pred)
print(f'Loss for test data:', curr_loss)
Y_test_pred_binarised = (Y_test_pred >= threshold_scaled)*1
print(Y_test_pred_binarised)

Y_train_pred = sigmoid_2d(X_scaled_train, W, b)
Y_train_pred_binarised = (Y_train_pred >= threshold_scaled)*1

print('Accuracy score test: ', accuracy_score(
    Y_binarised_test.ravel(), Y_test_pred_binarised)
)
print('Accuracy score train: ',
      accuracy_score(Y_binarised_train.ravel(), Y_train_pred_binarised))


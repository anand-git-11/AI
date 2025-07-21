import numpy as np
from lib_util import FNN
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(0)
if __name__ == '__main__':

    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["red", "yellow", "green"])

    data, labels_orig = make_blobs(
        n_samples=1000,
        centers=4,
        n_features=2,
        random_state=0
    )
    labels = np.mod(labels_orig, 2)
    # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
    # plt.show()

    X_train, X_val, Y_train, Y_val = train_test_split(
        data, labels, stratify=labels, random_state=0)
    print(X_train.shape, X_val.shape)

    # X = np.array([[1, 2], [3, 4]])
    # neurons_cnt_each_layer = [2, 1, 1]
    # fnn = FNN(X, neurons_cnt_each_layer)
    # print(X)
    # print('neurons_cnt_each_layer: ', neurons_cnt_each_layer)
    # print('\n\n')
    # print(fnn.W, '\n', fnn.B)
    mx_accuracy_train, mx_accuracy_val, mx_threshold = -1, -1, -1
    eta = 0
    for eta in np.arange(0.1, 1, 0.05):
        f = FNN()
        f.fit(
            X_train, Y_train,
            epochs=2000,
            learning_rate=eta,
            display_loss=False,
            initialise=True
        )
        for threshold in np.arange(0.0, 1, 0.1):
            print('\n', '*'*10, threshold, '*'*10)
            Y_pred_train = f.predict(X_train)
            print('negative')
            tmp = np.column_stack((Y_train, Y_pred_train))
            tmp1 = tmp[(tmp[:, 0] - tmp[:, 1]) < -threshold]
            print(tmp1.shape)
            if tmp1.shape[0]:
                print('mean: ', np.mean(tmp1))
            # for ele in tmp1:
            #     print(ele)
            print('positive')
            tmp1 = tmp[(tmp[:, 0] - tmp[:, 1]) > threshold]
            print(tmp1.shape)
            if tmp1.shape[0]:
                print('mean: ', np.mean(tmp1))
            # for ele in tmp1:
            #     print(ele)

            # plt.plot(Y_train, '', marker='*', label='Y_train')
            # plt.plot(Y_pred_train, '', marker='o', label='Y_pred_train')
            # plt.scatter(
            #     np.array(list(range(Y_train.shape[0]))), Y_train, marker='*')#, c=Y_train, cmap=my_cmap)
            # plt.scatter(
            #     np.array(list(range(Y_pred_train.shape[0]))), Y_pred_train, marker='o')#, c=Y_pred_train, cmap=my_cmap)
            # plt.legend()
            # plt.show()

            Y_pred_binarised_train = (Y_pred_train >= threshold).astype("int").ravel()
            Y_pred_val = f.predict(X_val)
            Y_pred_binarised_val = (Y_pred_val >= threshold).astype("int").ravel()

            accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
            accuracy_val = accuracy_score(Y_pred_binarised_val, Y_val)

            print("Training accuracy", round(accuracy_train, 2))
            print("Validation accuracy", round(accuracy_val, 2))

            if mx_accuracy_train < accuracy_train and mx_accuracy_val < accuracy_val:
                mx_accuracy_train, mx_accuracy_val = accuracy_train, accuracy_val
                mx_threshold = threshold
                eta = eta

    print('\n\n')
    print("Max Training accuracy", round(mx_accuracy_train, 2))
    print("Max Validation accuracy", round(mx_accuracy_val, 2))
    print("Max threshold", round(mx_threshold, 2))

# write your code here
import math

import numpy
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, lr=0.01, epochs=100):
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.epochs = epochs

    @staticmethod
    def sigmoid(t: numpy.array) -> float:
        """
        :param t:
        :return:
        """
        val = 1 / (1 + math.exp(-t))
        return val

    def predict_proba(self, row, cof_):
        """
        :param row:
        :param cof_:
        :return:
        """
        t = np.dot(cof_, row)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        row_count = X_train.shape[0]
        mse_error_first = [0.] * y_train.shape[0]
        mse_error_last = [0.] * y_train.shape[0]
        self.coef_ = np.array([0.] * len(X_train.keys())) if self.fit_intercept else np.array(
            [0.] * len(X_train.keys()))
        for epoch in range(self.epochs):
            for i in range(len(X_train)):
                row = X_train.iloc[i]
                y_hat = self.predict_proba(np.array(row), self.coef_)
                y_i = y_train.iloc[i]

                # Update the weights
                for index, train_row in enumerate(row):
                    if self.fit_intercept and index == 0:
                        # Update the bias b_0
                        self.coef_[index] += -self.lr * (y_hat - y_i) * y_hat * (1 - y_hat)
                        continue
                    if train_row != 'intercept':
                        x_i_j = row[index]
                        derive = self.lr * (y_hat - y_i) * y_hat * (1 - y_hat) * x_i_j
                        self.coef_[index] = self.coef_[index] - derive

                if epoch == 0:
                    mse_error_first[i] = ((y_hat - y_i) ** 2) / row_count

                if epoch == self.epochs - 1:
                    mse_error_last[i] = ((y_hat - y_i) ** 2) / row_count

        return mse_error_first, mse_error_last

    def fit_log_loss(self, X_train, y_train):
        row_count = X_train.shape[0]
        log_loss_error_first = [0.] * y_train.shape[0]
        log_loss_error_last = [0.] * y_train.shape[0]
        self.coef_ = np.array([0.] * len(X_train.keys())) if self.fit_intercept else np.array(
            [0.] * len(X_train.keys()))
        for epoch in range(self.epochs):
            for i in range(len(X_train)):

                row = X_train.iloc[i]
                y_hat = self.predict_proba(np.array(row), self.coef_)
                y_i = y_train.iloc[i]

                for index, train_row in enumerate(row):
                    if self.fit_intercept and index == 0:
                        self.coef_[index] += - ((self.lr * (y_hat - y_i)) / row_count)
                        continue
                    x_i_j = row[index]
                    derive = (self.lr * (y_hat - y_i) * x_i_j) / row_count
                    self.coef_[index] -= derive

                if epoch == 0:
                    log_loss_error_first[i] = -(y_i * math.log(y_hat) + (1 - y_i) * math.log(1 - y_hat)) / row_count

                if epoch == self.epochs - 1:
                    log_loss_error_last[i] = -(y_i * math.log(y_hat) + (1 - y_i) * math.log(1 - y_hat)) / row_count
        return log_loss_error_first, log_loss_error_last

    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for i in range(len(X_test)):
            y_hat = self.predict_proba(X_test.iloc[i], self.coef_)
            predictions.append(1) if y_hat >= cut_off else predictions.append(0)
        return np.array(predictions)  # predictions are binary values - 0 or 1


def standardize(feature):
    mean = np.mean(feature)
    std_deviation = np.std(feature)
    z = (feature - mean) / std_deviation
    return z


def intercept_is_true(X_train, X_test):
    X_test['intercept'] = 1.
    X_train['intercept'] = 1.
    x_test_col = X_test.pop('intercept')
    x_train_col = X_train.pop('intercept')
    X_train.insert(0, 'intercept', x_train_col)
    X_test.insert(0, 'intercept', x_test_col)

    return X_test, X_train



def main():
    df_features, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = df_features[['worst concave points', 'worst perimeter', 'worst radius']]
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=43)
    # Creating  LogisiticRegression with different loss functions each
    model_2 = LogisticRegression()
    model_1 = CustomLogisticRegression(fit_intercept=True, lr=0.01, epochs=1000)
    model_0 = CustomLogisticRegression(fit_intercept=True, lr=0.01, epochs=1000)
    if model_0.fit_intercept:
        X_test, X_train = intercept_is_true(X_test=X_test, X_train=X_train)

    # Getting the mse error from the last and first epochs
    mse_error_first, mse_error_last = model_1.fit_mse(X_train, y_train)
    # Same for the log_loss to later compare the models
    log_loss_error_first, log_loss_error_last = model_0.fit_log_loss(X_train, y_train)

    predictions_mse = model_1.predict(X_test=X_test)

    predictions_log_loss = model_0.predict(X_test=X_test)
    mse_acc = accuracy_score(y_true=y_test, y_pred=predictions_mse)
    log_loss_acc = accuracy_score(y_true=y_test, y_pred=predictions_log_loss)

    model_2.fit(X_train, y_train)
    sklearn_prediction = model_2.predict(X_test)
    sklearn_acc = accuracy_score(y_true=y_test, y_pred=sklearn_prediction)

    dic = {'mse_accuracy': mse_acc, 'logloss_accuracy': log_loss_acc, 'sklearn_accuracy': sklearn_acc,
           'mse_error_first': mse_error_first,
           'mse_error_last': mse_error_last, 'logloss_error_first': log_loss_error_first,
           'logloss_error_last': log_loss_error_last}

    print(dic)

if __name__ == "__main__":
    main()

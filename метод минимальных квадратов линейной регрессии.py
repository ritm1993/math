import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
diabetes = load_diabetes()
scaler = MinMaxScaler()

inputs = scaler.fit_transform(diabetes.data)
targets = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3, random_state=42)


class LinearRegression:
  def __init__(self):
    self.weights = None

  def fit(self, X, y):
    X = np.insert(X, 0, 1, axis=1)
    self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
    return self

  def predict(self, X):
    X = np.insert(X, 0, 1, axis=1)
    return X @ self.weights


def mean_squared_error(y_true, y_pred):
  return np.mean((y_true - y_pred) ** 2)
model = LinearRegression().fit(X_train, y_train)
print(f'MSE модели на обучающей выборке {mean_squared_error(model.predict(X_train), y_train)}')
print(f'MSE модели на тестовой выборке {mean_squared_error(model.predict(X_test), y_test)}')


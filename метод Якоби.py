import numpy as np

# Создаем матрицу коэффициентов
A = np.array([[4, 2, -1, -1],
              [2, 5, 1, -1],
              [-1, 1, 3, -8],
              [-1, -1, 2, 2]])

# Создаем вектор правой части
b = np.array([1, 1, 1, 1])

# Создаем начальное приближение
x0 = np.array([0, 0, 0, 0])

# Максимальное число итераций
max_iter = 1000

# Задаем точность вычислений
eps = 0.01

# Получаем размеры матрицы A
n, m = A.shape

# Создаем диагональную матрицу D и матрицу R
D = np.zeros((n, n))
R = np.zeros((n, n))

# Заполняем матрицы D и R
for i in range(n):
    for j in range(n):
        if i == j:
            D[i][j] = A[i][j]
        else:
            R[i][j] = A[i][j]

# Вычисляем обратную диагональную матрицу D_inv
D_inv = np.linalg.inv(D)

# Создаем матрицу T и вектор c
T = -D_inv.dot(R)
c = D_inv.dot(b)

# Начинаем итерационный процесс
x = x0
for i in range(max_iter):
    x_new = T.dot(x) + c
    if np.linalg.norm(x_new - x) < eps:
        break
    x = x_new

print("Solution:")
print(x)

import math
import copy

# Пробные данные для уравнения A*X = B
# x = [1.10202, 0.99091, 1.01111]

a = [[10, 1, -1],
     [1, 10, -1],
     [-1, 1, 10]]

b = [11, 10, 10]


# Проверка матрицы коэффициентов на корректность
def isCorrectArray(a):
  for row in range(0, len(a)):
    if (len(a[row]) != len(b)):
      print('Не соответствует размерность')
      return False

  for row in range(0, len(a)):
    if (a[row][row] == 0):
      print('Нулевые элементы на главной диагонали')
      return False
  return True


# Условие завершения программы на основе вычисления
# расстояния между соответствующими элементами соседних
# итераций в методе решения
def isNeedToComplete(x_old, x_new):
  eps = 0.0001
  sum_up = 0
  sum_low = 0
  for k in range(0, len(x_old)):
    sum_up += (x_new[k] - x_old[k]) ** 2
    sum_low += (x_new[k]) ** 2

  return math.sqrt(sum_up / sum_low) < eps


# Процедура решения
def solution(a, b):
  if (not isCorrectArray(a)):
    print('Ошибка в исходных данных')
  else:
    count = len(b)  # количество корней

    x = [1 for k in range(0, count)]  # начальное приближение корней

    numberOfIter = 0  # подсчет количества итераций
    MAX_ITER = 100  # максимально допустимое число итераций
    while (numberOfIter < MAX_ITER):

      x_prev = copy.deepcopy(x)

      for k in range(0, count):
        S = 0
        for j in range(0, count):
          if (j != k): S = S + a[k][j] * x[j]
        x[k] = b[k] / a[k][k] - S / a[k][k]

      if isNeedToComplete(x_prev, x):  # проверка на выход
        break

      numberOfIter += 1

    print('Количество итераций на решение: ', numberOfIter)

    return x

  # MAIN - блок программмы


print('Решение: ', solution(a, b))  # Вызываем процедуру решение

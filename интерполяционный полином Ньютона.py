def newton_interpolation(x_values, y_values, x):
  """
  :param x_values: список значений x
  :param y_values: список значений y
  :param x: значение, для которого нужно найти приближенное значение y
  :return: приближенное значение y, рассчитанное с помощью интерполяционного полинома Ньютона
  """
  n = len(x_values)
  # Инициализация разделенных разностей
  f = [[None] * n for _ in range(n)]
  for i in range(n):
    f[i][0] = y_values[i]
  # Вычисление разделенных разностей
  for j in range(1, n):
    for i in range(n - j):
      f[i][j] = (f[i + 1][j - 1] - f[i][j - 1]) / (x_values[i + j] - x_values[i])
  # Вывод таблицы конечных разностей
  print("Таблица конечных разностей:")
  for row in f:
    for elem in row:
      if elem is not None:
        print("{:.4f}".format(elem), end="\t")
      else:
        print("\t", end="")
    print()
  # Вычисление значения интерполяционного полинома
  y = 0
  for i in range(n):
    prod = f[0][i]
    for j in range(i):
      prod *= (x - x_values[j])
    y += prod
  return y


x_values = [1, 2, 3, 5, 7, 8, 11, 14, 15]
y_values = [-3, 3, 5, 4, 1, -5, 6, 3, 2]
x = 1
y = newton_interpolation(x_values, y_values, x)
print("Значение интерполяционного полинома в точке x = {:.2f}: {:.4f}".format(x, y))
Интерполяционный полином Лагранжа.
import numpy as np

def lagrange_interpolation(x, y):
  def basis_polynomial(i, x):
    xi = x[i]
    return np.prod([(x - x[j]) / (xi - x[j]) for j in range(len(x)) if j != i])

  return sum(y[i] * basis_polynomial(i, x) for i in range(len(x)))


x_points = np.array([1, 2, 3, 5, 7, 8, 11, 14, 15])
y_points = np.array([-3, 3, 5, 4, 1, -5, 6, 3, 2])
x = 8

result = lagrange_interpolation(x_points, y_points)
print(f"Значение функции в точке x={x}: {result}")

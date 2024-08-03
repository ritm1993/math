# import math
#
#
# class Element:
#   def __init__(self, numL, numR):
#     self.l = numL
#     self.r = numR
#
#   def __add__(self, other):
#     return Element(self.l + other.l, self.r + other.r)
#
#   def __sub__(self, other):
#     return Element(self.l - other.r, self.r - other.l)
#
#   def __mul__(self, other):
#     t1 = self.l * other.l
#     t2 = self.l * other.r
#     m1 = self.r * other.l
#     m2 = self.r * other.r
#     return Element(min(t1, t2), max(m1, m2))
#
#   def __truediv__(self, other):
#     t1 = self.l / other.l
#     t2 = self.l / other.r
#     m1 = self.r / other.l
#     m2 = self.r / other.r
#     return Element(min(t1, t2), max(m1, m2))
#
#
# def initial(a, b, c, d, M, R):
#   for i in range(M):
#     a.append(Element(0, 0))
#     b.append(Element(0, 0))
#     c.append(Element(0, 0))
#     d.append(Element(0, 0))
#
#   for i in range(1, M):
#     a[i] = Element((0.3 * math.sin(i + 1) / V) - R, (0.3 * math.sin(i + 1) / V) + R)
#
#   for i in range(M):
#     b[i] = Element(((10 * V) + ((i + 1) / V)) - R, ((10 * V) + ((i + 1) / V)) + R)
#     d[i] = Element((1.3 + (i + 1) / V) - R, (1.3 + (i + 1) / V) + R)
#
#   for i in range(M - 1):
#     c[i] = Element((0.4 * math.cos(i + 1) / V) - R, (0.4 * math.cos(i + 1) / V) + R)
#
#   return a, b, c, d
#
#
# def progonka(a, b, c, d, M):
#   alpha = []
#   beta = []
#   x = []
#   for i in range(M):
#     x.append(Element(0, 0))
#
#   alpha.append(Element(-1, -1) * c[0] / b[0])
#   beta.append(d[0] / b[0])
#
#   for i in range(1, M):
#     alpha.append((Element(-1, -1) * c[i]) / (b[i] + a[i] * alpha[i - 1]))
#     beta.append((d[i] - a[i] * beta[i - 1]) / (b[i] + a[i] * alpha[i - 1]))
#
#   x[M - 1] = beta[M - 1]
#   for i in reversed(range(M - 1)):
#     x[i] = alpha[i] * x[i + 1] + beta[i]
#
#   return alpha, beta, x
#
#
# def showEl(arrayEl):
#   for e in arrayEl:
#     print("{:.1e}".format((e.l + e.r) / 2))
#
#
# def showX(arrayX):
#   for e in arrayX:
#     print("{:.5e}    {:.5e}   {:.5e}".format(e.l, (e.l + e.r) / 2, e.r))
#
#
# if (__name__ == '__main__'):
#   R = 0.01  # отклонение
#   M = int(input("Введите размерность: "))
#   V = 25.0
#   print("Вариант %s: " % (V))
#
#   '''
#    c - диагональ, лежащая над главной (нумеруется: [0;n-2])
#    b - главная диагональ матрицы A (нумеруется: [0;n-1])
#    a - диагональ, лежащая под главной (нумеруется: [1;n-1])
#    d - правая часть (столбец)
#   '''
#   a = [2, 4, 2]
#   c = [1, -1]
#   b = [0, 3, 2]
#   d = [3, 7, -3, -2]
#   a, b, c, d = initial(a, b, c, d, M, R)
#
# alpha, beta, x = progonka(a, b, c, d, M)
#
#     print("\na")
#     showEl(a)
#
#     print("\nb")
#     showEl(b)
#
#     print("\nc")
#     showEl(c)
#
#     print("\nd")
#     showEl(d)
#
#     print("\nalpha")
#     showEl(alpha)
#
#     print("\nbeta")
#     showEl(beta)
#
#     print("\nx")
#     showX(x)
import numpy as np

h = 0.1
n = 5  # число интервалов
x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])  # число точек n+1
bn = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
p = x * x
q = 1.8 - x
for b in bn:
  f = x / (x * x + b)
  m = -2 + h * p
  k = 1 - h * p + h * h * q
  y = np.zeros(n + 1)
  c = np.zeros(n)
  d = np.zeros(n)
  alpha0 = 1;
  beta0 = 1
  alpha1 = 0;
  beta1 = 0
  A, B = 0, 0
  d[0] = (k[0] * A * h / (alpha1 - alpha0 * h) + f[0] * h * h)
  c[0] = (alpha1 - alpha0 * h) / (m[0] * (alpha1 - alpha0 * h) + k[0] * alpha1)

  for i in range(1, n - 1):
    c[i] = 1 / (m[i] - k[i] * c[i - 1])
    d[i] = f[i] * h * h - k[i] * c[i - 1] * d[i - 1]
  y[n] = (beta1 * c[n - 2] + B * h) / (beta1 * (1 + c[n - 2]) + beta0 * h)

  for i in range(n - 2, -1, -1):
    y[i + 1] = c[i] * (d[i] - y[i + 2 - 1])
  y[0] = (alpha1 * y[1] - A * h) / (alpha1 - alpha0 * h)
  print(b, y)


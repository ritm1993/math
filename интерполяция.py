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



# import numpy as np
# import matplotlib.pyplot as plt
#
# def quadratic_approximation(x, y):
#     # Создаем матрицу A для нашей системы уравнений
#     A = np.vstack([x**2, x, np.ones(len(x))]).T
#     # Вычисляем вектор коэффициентов
#     a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
#     # Возвращаем функцию вида ax^2 + bx + c
#     return lambda x: a*x**2 + b*x + c
#
# # Задаем нашу функцию
# x = np.linspace(-5, 5, num=50)
# y = x**2 + np.random.normal(0, 1, len(x))
#
# # Аппроксимируем функцию
# f = quadratic_approximation(x, y)
#
# # Строим графики исходной функции и аппроксимации
# plt.plot(x, y, 'o', label='Data')
# plt.plot(x, f(x), label='Quadratic approximation')
# plt.legend()
# plt.show()
# import numpy as np
#
# A=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,0,0]])
# B = np.array([1,1,1,1,1])
#
# W = np.array([1,2,3,4,5])
#
# Aw = A * np.sqrt(W[:,np.newaxis])
# Bw = B * np.sqrt(W)
#
# X = np.linalg.lstsq(Aw, Bw, rcond = -1)
# print(X[0])
#
#
#
# # import numpy as np
# #
# # def lagrange_interpolation(x, y):
# #   def basis_polynomial(i, x):
# #     xi = x[i]
# #     return np.prod([(x - x[j]) / (xi - x[j]) for j in range(len(x)) if j != i])
# #
# #   return sum(y[i] * basis_polynomial(i, x) for i in range(len(x)))
# #
# #
# # x_points = np.array([1, 2, 3, 5, 7, 8, 11, 14, 15])
# # y_points = np.array([-3, 3, 5, 4, 1, -5, 6, 3, 2])
# # x = 8
# #
# # result = lagrange_interpolation(x_points, y_points)
# # print(f"Значение функции в точке x={x}: {result}")
#
#
#
#
# # import numpy as np
# # from sympy import *
# # import matplotlib.pyplot as plt
# #
# #
# # def f(x):
# #     return 1 / (1 + x ** 2)
# #
# #
# # def cal(begin, end, i):
# #     by = f(begin)
# #     ey = f(end)
# #     I = Ms[i] * ((end - n) ** 3) / 6 + Ms[i + 1] * ((n - begin) ** 3) / 6 + (by - Ms[i] / 6) * (end - n) + (
# #             ey - Ms[i + 1] / 6) * (n - begin)
# #     return I
# #
# #
# # def ff(x):  # f[x0, x1, ..., xk]
# #     ans = 0
# #     for i in range(len(x)):
# #         temp = 1
# #         for j in range(len(x)):
# #             if i != j:
# #                 temp *= (x[i] - x[j])
# #         ans += f(x[i]) / temp
# #     return ans
# #
# #
# # def calM():
# #     lam = [1] + [1 / 2] * 9
# #     miu = [1 / 2] * 9 + [1]
# #     Y = 1 / (1 + n ** 2)
# #     df = diff(Y, n)
# #     x = np.array(range(11)) - 5
# #     ds = [6 * (ff(x[0:2]) - df.subs(n, x[0]))]
# #     # ds = [6 * (ff(x[0:2]) - 1)]
# #     for i in range(9):
# #         ds.append(6 * ff(x[i: i + 3]))
# #     ds.append(6 * (df.subs(n, x[10]) - ff(x[-2:])))
# #     # ds.append(6 * (1 - ff(x[-2:])))
# #     Mat = np.eye(11, 11) * 2
# #     for i in range(11):
# #         if i == 0:
# #             Mat[i][1] = lam[i]
# #         elif i == 10:
# #             Mat[i][9] = miu[i - 1]
# #         else:
# #             Mat[i][i - 1] = miu[i - 1]
# #             Mat[i][i + 1] = lam[i]
# #     ds = np.mat(ds)
# #     Mat = np.mat(Mat)
# #     Ms = ds * Mat.I
# #     return Ms.tolist()[0]
# #
# #
# # def calnf(x):
# #     nf = []
# #     for i in range(len(x) - 1):
# #         nf.append(cal(x[i], x[i + 1], i))
# #     return nf
# #
# #
# # def calf(f, x):
# #     y = []
# #     for i in x:
# #         y.append(f.subs(n, i))
# #     return y
# #
# #
# # def nfSub(x, nf):
# #     tempx = np.array(range(11)) - 5
# #     dx = []
# #     for i in range(10):
# #         labelx = []
# #         for j in range(len(x)):
# #             if x[j] >= tempx[i] and x[j] < tempx[i + 1]:
# #                 labelx.append(x[j])
# #             elif i == 9 and x[j] >= tempx[i] and x[j] <= tempx[i + 1]:
# #                 labelx.append(x[j])
# #         dx = dx + calf(nf[i], labelx)
# #     return np.array(dx)
# #
# #
# # def draw(nf):
# #     plt.rcParams['font.sans-serif'] = ['SimHei']
# #     plt.rcParams['axes.unicode_minus'] = False
# #     x = np.linspace(-5, 5, 101)
# #     y = f(x)
# #     Ly = nfSub(x, nf)
# #     plt.plot(x, y, label='Примитивный')
# #     plt.plot(x, Ly, label='Функция интерполяции кубическим сплайном')
# #     plt.xlabel('x')
# #     plt.ylabel('y')
# #     plt.legend()
# #
# #     plt.savefig('1.png')
# #     plt.show()
# #
# #
# # def lossCal(nf):
# #     x = np.linspace(-5, 5, 101)
# #     y = f(x)
# #     Ly = nfSub(x, nf)
# #     Ly = np.array(Ly)
# #     temp = Ly - y
# #     temp = abs(temp)
# #     print(temp.mean())
# #
# #
# # if __name__ == '__main__':
# #     x = np.array(range(11)) - 5
# #     y = f(x)
# #
# #     n, m = symbols('n m')
# #     init_printing(use_unicode=True)
# #     Ms = calM()
# #     nf = calnf(x)
# #     draw(nf)
# #     lossCal(nf)
#
# # def DFTMatrix(nodesNumber):
# #   ''' computes matrix W as defined in Eq. (5.2) of the script
# #
# #       Parameter:
# #       nodesNumber (int): number of nodes
# #
# #       Returns:
# #       W (array): discrete Fourier Transform matrix where W is
# #                  a square matrix with dimension nodesNumber x nodesNumber
# #   '''
# #
# #   W1 = np.zeros([nodesNumber, nodesNumber], dtype=complex)  # right triangle matrix
# #   for l in range(nodesNumber):
# #     for k in range(l, nodesNumber):
# #       b = 2 * pi * l * k / nodesNumber
# #       W1[l][k] = exp(b * 1j)
# #
# #   W2 = np.copy(W1.T)  # left triangle matrix
# #
# #   for j in range(nodesNumber):  # delete entries on diagonal
# #     W2[j][j] = 0
# #
# #   W = W1 + W2
# #
# #   return W
# #
# #
# # def DFT(f_nodes, fourierMat):
# #   ''' computes discrete Fourier Transform with complex coefficients
# #
# #       Parameters:
# #       f_nodes (array): values of function in node points with dimension N (number of nodes)
# #       fourierMat (array): matrix as in DFTMatrix with dimension N x N
# #
# #       Returns:
# #       fourierF (array): complex values of discrete Fourier Transformation with dimension N
# #       A, B (array): real coefficients of trigonometric polynomial which interpolates f (check p.42 in script)
# #                     with dimension N-1/2 resp. N/2
# #   '''
# #   N = len(fourierMat)
# #
# #   A = np.zeros([int(N / 2) + 1])
# #   B = np.zeros([int(N / 2)])
# #
# #   fourierF = fourierMat.dot(f_nodes)
# #
# #   A[0] = fourierF[0].real
# #
# #   for m in range(1, int(N / 2)):
# #     A[m] = -2 * fourierF[m].real
# #     B[m] = -2 * fourierF[m].imag
# #
# #   if N % 2 == 0:
# #     A[int(N / 2)] = fourierF[int(N / 2)].real
# #   else:
# #     A[int(N / 2)] = 0
# #
# #   return fourierF, A, B
# #
# #
# # def funct(ev_pts):
# #   ''' computes f(x) = |x| for x in ev_pts
# #
# #       Parameters:
# #       ev_pts (array): evaluation points
# #
# #       Returns:
# #       f_eval (array): values of f in evaluation points
# #   '''
# #   f_eval = [np.abs(x) for x in ev_pts]
# #
# #   return f_eval
# #
# #
# # def get_trigoInterpoly(ev_pts, A, B, N):
# #   ''' computes trigonometric interpolation of f(x) = |x| as seen on page 41 in the script
# #
# #       Parameters.
# #       ev_pts (array): evaluation pts of f in some interval [-pi,pi]
# #       A, B (array): DFT coefficients from the function DFT
# #       nodesNumber (int): number of nodes where we interpolate f
# #
# #       Returns:
# #       T (array): trigonometric interpolation of f in interval [a,b]
# #   '''
# #
# #   def T(t):
# #     ''' evaluates the trigonometric sum defined by A, B in the point ev_pt
# #
# #         Parameters:
# #         t (float): evaluation point
# #         A, B (array): real coefficients from DTF
# #         N (int): number of nodes
# #
# #     '''
# #     trigsum = A[0]
# #     for m in range(1, int(N / 2)):
# #       trigsum += A[m] * cos(m * t) + B[m] * sin(m * t)
# #
# #     if N % 2 == 0:
# #       trigsum += A[int(N / 2)]
# #
# #     return trigsum
# #
# #   trigo_interpoly = [(1 / N) * T(t) for t in ev_pts]
# #
# #   return trigo_interpoly
# #
# #
# # def plot_DFT():
# #   ''' plots function f(x) = |x| and the DFT of f
# #   '''
# #   ev_pts = np.linspace(-pi, pi, 1000)
# #   f_eval = funct(ev_pts)
# #   plt.figure()
# #   plt.plot(ev_pts, f_eval, c='black', label='|x|')
# #
# #   for n in [2, 4, 8, 16]:
# #     nodes = np.linspace(-pi, pi, n)
# #     f_nodes = funct(nodes)
# #     fourierM = DFTMatrix(n)
# #     fourierF, A, B = DFT(f_nodes, fourierM)
# #     trigo_interpoly = get_trigoInterpoly(f_eval, A, B, n)
# #     plt.plot(ev_pts, trigo_interpoly, label='T(t) for n={}'.format(n))
# #
# #   plt.show()
# # Представляем библиотечные функции; определяем константы
# # import math
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # pi = math.pi
# #
# #
# # # БПФ Шаг 1. Представляем массив комплексных чисел в A1
# # def init_A(Interpolation_times, f, func, l, r):
# #   point_num = 2 * Interpolation_times
# #   A = []
# #   interval = 2 * pi / point_num
# #   now = -pi
# #   for i in range(point_num):
# #     A.append(complex(func(f, l, r, now), 0))
# #     now += interval
# #   return A
#
#
# # # FFT Шаг 2: Предварительная обработка единичного корня (каждый узел интерполяции)
# # def init_w(Interpolation_times):
# #   point_num = 2 * Interpolation_times
# #   w = []
# #   w.append(complex(1, 0))
# #   degree = 2 * pi / point_num
# #   root = complex(math.cos(degree), math.sin(degree))
# #   for i in range(1, Interpolation_times):
# #     w.append(w[i - 1] * root)
# #   return w
#
#
# # # Шаги 4-8 БПФ: ДПФ
# # def DFT(A1, w, Interpolation_times, p):
# #   point_num = 2 * Interpolation_times
# #   A2 = [complex(0, 0)]
# #   A2 = A2 * point_num
# #   for q in range(1, p + 1):
# #     if q % 2 == 1:
# #       for k in range(2 ** (p - q)):
# #         for j in range(0, 2 ** (q - 1)):
# #           tmp1 = k * 2 ** q + j
# #           tmp2 = k * 2 ** (q - 1) + j
# #           tmp3 = 2 ** (p - 1)
# #           tmp4 = 2 ** (q - 1)
# #           A2[tmp1] = A1[tmp2] + A1[tmp2 + tmp3]
# #           A2[tmp1 + tmp4] = (A1[tmp2] - A1[tmp2 + tmp3]) * w[k * tmp4]
# #     else:
# #       for k in range(2 ** (p - q)):
# #         for j in range(0, 2 ** (q - 1)):
# #           tmp1 = k * 2 ** q + j
# #           tmp2 = k * 2 ** (q - 1) + j
# #           tmp3 = 2 ** (p - 1)
# #           tmp4 = 2 ** (q - 1)
# #           A1[tmp1] = A2[tmp2] + A2[tmp2 + tmp3]
# #           A1[tmp1 + tmp4] = (A2[tmp2] - A2[tmp2 + tmp3]) * w[k * tmp4]
# #   c = []
# #   if p % 2 == 0:
# #     for i in range(Interpolation_times + 1):
# #       c.append(A1[i])
# #   else:
# #     for i in range(Interpolation_times + 1):
# #       c.append(A2[i])
# #   return c
# #
# #
# # # Восстановить c до a и b
# # def exchange(c, Interpolation_times):
# #   root = complex(-1, 0)
# #   now_degree = complex(1, 0)
# #   a = []
# #   b = []
# #   for i in range(Interpolation_times + 1):
# #     ci = c[i] * now_degree
# #     a.append(ci.real / Interpolation_times)
# #     b.append(ci.imag / Interpolation_times)
# #     now_degree *= root
# #   return a, b
# #
# #
# # # БПФ весь процесс, сводка вышеуказанных шагов
# # def FFT(Interpolation_times, p, f, func, l, r):
# #   a = []
# #   b = []
# #   A1 = init_A(Interpolation_times, f, func, l, r)
# #   w = init_w(Interpolation_times)
# #   c = DFT(A1, w, Interpolation_times, p)
# #   a, b = exchange(c, Interpolation_times)
# #   return a, b
# #
# #
# # # Вычислить значение точки, соответствующее тригонометрическому полиному
# # def ans_F(x, a, b, Interpolation_times):
# #   res = a[0] / 2
# #   for i in range(1, Interpolation_times + 1):
# #     res += a[i] * math.cos(i * x) + b[i] * math.sin(i * x)
# #   return res
# #
# #
# # # Функция рисования: нарисуйте исходную функцию и встроенную функцию отдельно
# # def draw_pic(a, b, Interpolation_times, f, func, l, r):
# #   x = np.arange(-pi, pi, 0.01)
# #   y = []
# #   yy = []
# #   for i in range(len(x)):
# #     y.append(ans_F(x[i], a, b, Interpolation_times))
# #     yy.append(func(f, l, r, x[i]))
# #     x[i] = (x[i] * (r - l)) / (2 * pi) + (l + r) / 2
# #   fig = plt.figure()
# #   plt.plot(x, y, label='interpolation')
# #   plt.plot(x, yy, label='raw')
# #   plt.legend()
# #   plt.show()
# #   plt.close(fig)
# #
# #
# # # Распечатать результаты примерки
# # def judge_sign(a):
# #   if a < 0:
# #     return '-'
# #   else:
# #     return '+'
# #
# #
# # def print_trans_result(a, b, Interpolation_times, l, r):
# #   print("S(y) = %f" % (a[0] / 2))
# #   print("       %c %.3lf cos(y) %c %.3lf sin(y)"
# #         % (judge_sign(a[1]), abs(a[1]), judge_sign(b[1]), abs(b[1])))
# #   for i in range(2, Interpolation_times + 1):
# #     print("       %c %.3lf cos(%dy) %c %.3lf sin(%dy)"
# #           % (judge_sign(a[i]), abs(a[i]), i, judge_sign(b[i]), abs(b[i]), i))
# #
# #   print("")
# #   tmp = pi * (l + r) / (r - l)
# #   print("y = %.3lfx %c %.3lf"
# #         % (2 * pi / (r - l), judge_sign(-tmp), abs(tmp)))
# #
# #
# # # Обработка для заданного количества интерполяций, не являющегося целой степенью 2
# # # Мы увеличиваем количество точек интерполяции до ближайшей целой степени 2 и обрезаем конец
# # def extend(Interpolation_times):
# #   lim = 1
# #   p = 0
# #   while lim < Interpolation_times:
# #     lim *= 2
# #     p += 1
# #   return lim, p
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # def quadratic_approximation(x, y):
# #     # Создаем матрицу A для нашей системы уравнений
# #     A = np.vstack([x**2, x, np.ones(len(x))]).T
# #     # Вычисляем вектор коэффициентов
# #     a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
# #     # Возвращаем функцию вида ax^2 + bx + c
# #     return lambda x: a*x**2 + b*x + c
# #
# # # Задаем нашу функцию
# # x = np.linspace(-5, 5, num=50)
# # y = x**2 + np.random.normal(0, 1, len(x))
# #
# # # Аппроксимируем функцию
# # f = quadratic_approximation(x, y)
# #
# # # Строим графики исходной функции и аппроксимации
# # plt.plot(x, y, 'o', label='Data')
# # plt.plot(x, f(x), label='Quadratic approximation')
# # plt.legend()
# # plt.show()
# # import numpy as np
# #
# # A=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,0,0]])
# # B = np.array([1,1,1,1,1])
# #
# # W = np.array([1,2,3,4,5])
# #
# # Aw = A * np.sqrt(W[:,np.newaxis])
# # Bw = B * np.sqrt(W)
# #
# # X = np.linalg.lstsq(Aw, Bw, rcond = -1)
# # print(X[0])
# # import bisect
# # import math
# #
# # import numpy as np
# #
# #
# # class Spline:
# #     """
# #     Cubic Spline class
# #     """
# #
# #     def __init__(self, x, y):
# #         self.b, self.c, self.d, self.w = [], [], [], []
# #
# #         self.x = x
# #         self.y = y
# #
# #         self.nx = len(x)  # dimension of x
# #         h = np.diff(x)
# #
# #         # calc coefficient c
# #         self.a = [iy for iy in y]
# #
# #         # calc coefficient c
# #         A = self.__calc_A(h)
# #         B = self.__calc_B(h)
# #         self.c = np.linalg.solve(A, B)
# #
# #         # calc spline coefficient b and d
# #         for i in range(self.nx - 1):
# #             self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
# #             tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
# #                  (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
# #             self.b.append(tb)
# #
# #     def calc(self, t):
# #         """
# #         Calc position
# #         if t is outside of the input x, return None
# #         """
# #
# #         if t < self.x[0]:
# #             return None
# #         elif t > self.x[-1]:
# #             return None
# #
# #         i = self.__search_index(t)
# #         dx = t - self.x[i]
# #         result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
# #
# #         return result
# #
# #     def calc_d(self, t):
# #         """
# #         Calc first derivative
# #         if t is outside of the input x, return None
# #         """
# #
# #         if t < self.x[0]:
# #             return None
# #         elif t > self.x[-1]:
# #             return None
# #
# #         i = self.__search_index(t)
# #         dx = t - self.x[i]
# #         result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
# #         return result
# #
# #     def calc_dd(self, t):
# #         """
# #         Calc second derivative
# #         """
# #
# #         if t < self.x[0]:
# #             return None
# #         elif t > self.x[-1]:
# #             return None
# #
# #         i = self.__search_index(t)
# #         dx = t - self.x[i]
# #         result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
# #         return result
# #
# #     def __search_index(self, x):
# #         return bisect.bisect(self.x, x) - 1
# #
# #     def __calc_A(self, h):
# #         A = np.zeros((self.nx, self.nx))
# #         A[0, 0] = 1.0
# #         for i in range(self.nx - 1):
# #             if i != (self.nx - 2):
# #                 A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
# #             A[i + 1, i] = h[i]
# #             A[i, i + 1] = h[i]
# #
# #         A[0, 1] = 0.0
# #         A[self.nx - 1, self.nx - 2] = 0.0
# #         A[self.nx - 1, self.nx - 1] = 1.0
# #         #  print(A)
# #         return A
# #
# #     def __calc_B(self, h):
# #         """
# #         calc matrix B for spline coefficient c
# #         """
# #         B = np.zeros(self.nx)
# #         for i in range(self.nx - 2):
# #             B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
# #                        h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
# #         #  print(B)
# #         return B
# #
# #     def calc_curvature(self, t):
# #         j = int(math.floor(t))
# #         if j < 0:
# #             j = 0
# #         elif j >= len(self.a):
# #             j = len(self.a) - 1
# #
# #         dt = t - j
# #         df = self.b[j] + 2.0 * self.c[j] * dt + 3.0 * self.d[j] * dt * dt
# #         ddf = 2.0 * self.c[j] + 6.0 * self.d[j] * dt
# #         k = ddf / ((1 + df ** 2) ** 1.5)
# #         return k
# #
# #
# # class Spline2D:
# #     """
# #     2D Cubic Spline class
# #     """
# #
# #     def __init__(self, x, y):
# #         self.s = self.__calc_s(x, y)
# #         self.sx = Spline(self.s, x)
# #         self.sy = Spline(self.s, y)
# #
# #     def __calc_s(self, x, y):
# #         dx = np.diff(x)
# #         dy = np.diff(y)
# #         self.ds = [math.sqrt(idx ** 2 + idy ** 2)
# #                    for (idx, idy) in zip(dx, dy)]
# #         s = [0.0]
# #         s.extend(np.cumsum(self.ds))
# #         return s
# #
# #     def calc_position(self, s):
# #         """
# #         calc position
# #         """
# #         x = self.sx.calc(s)
# #         y = self.sy.calc(s)
# #
# #         return x, y
# #
# #     def calc_curvature(self, s):
# #         """
# #         calc curvature
# #         """
# #         dx = self.sx.calc_d(s)
# #         ddx = self.sx.calc_dd(s)
# #         dy = self.sy.calc_d(s)
# #         ddy = self.sy.calc_dd(s)
# #         k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5
# #         return k
# #
# #     def calc_yaw(self, s):
# #         """
# #         calc yaw
# #         """
# #         dx = self.sx.calc_d(s)
# #         dy = self.sy.calc_d(s)
# #         yaw = math.atan2(dy, dx)
# #         return yaw
# #
# #
# # def calc_2d_spline_interpolation(x, y, num=100):
# #     """
# #     Calc 2d spline course with interpolation
# #     :param x: interpolated x positions
# #     :param y: interpolated y positions
# #     :param num: number of path points
# #     :return:
# #         - x     : x positions
# #         - y     : y positions
# #         - yaw   : yaw angle list
# #         - k     : curvature list
# #         - s     : Path length from start point
# #     """
# #     sp = Spline2D(x, y)
# #     s = np.linspace(0, sp.s[-1], num+1)[:-1]
# #
# #     r_x, r_y, r_yaw, r_k = [], [], [], []
# #     for i_s in s:
# #         ix, iy = sp.calc_position(i_s)
# #         r_x.append(ix)
# #         r_y.append(iy)
# #         r_yaw.append(sp.calc_yaw(i_s))
# #         r_k.append(sp.calc_curvature(i_s))
# #
# #     travel = np.cumsum([np.hypot(dx, dy) for dx, dy in zip(np.diff(r_x), np.diff(r_y))]).tolist()
# #     travel = np.concatenate([[0.0], travel])
# #
# #     return r_x, r_y, r_yaw, r_k, travel
# #
# #
# # def test_spline2d():
# #     print("Spline 2D test")
# #     import matplotlib.pyplot as plt
# #     input_x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
# #     input_y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
# #
# #     x, y, yaw, k, travel = calc_2d_spline_interpolation(input_x, input_y, num=200)
# #
# #     plt.subplots(1)
# #     plt.plot(input_x, input_y, "xb", label="input")
# #     plt.plot(x, y, "-r", label="spline")
# #     plt.grid(True)
# #     plt.axis("equal")
# #     plt.xlabel("x[m]")
# #     plt.ylabel("y[m]")
# #     plt.legend()
# #
# #     plt.subplots(1)
# #     plt.plot(travel, [math.degrees(i_yaw) for i_yaw in yaw], "-r", label="yaw")
# #     plt.grid(True)
# #     plt.legend()
# #     plt.xlabel("line length[m]")
# #     plt.ylabel("yaw angle[deg]")
# #
# #     plt.subplots(1)
# #     plt.plot(travel, k, "-r", label="curvature")
# #     plt.grid(True)
# #     plt.legend()
# #     plt.xlabel("line length[m]")
# #     plt.ylabel("curvature [1/m]")
# #
# #     plt.show()
# #
# #
# # def test_spline():
# #     print("Spline test")
# #     import matplotlib.pyplot as plt
# #     x = [1,2,3,5,7,8,11,14,15]
# #     y = [-3,3,5,4,1,-5,6,3,2]
# #
# #     spline = Spline(x, y)
# #     rx = np.arange(-2.0, 4, 0.01)
# #     ry = [spline.calc(i) for i in rx]
# #
# #     plt.plot(x, y, "xb")
# #     plt.plot(rx, ry, "-r")
# #     plt.grid(True)
# #     plt.axis("equal")
# #     plt.show()
# #
# #
# # if __name__ == '__main__':
# #     test_spline()
# #     test_spline2d()
# #
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from scipy.interpolate import CubicSpline, interp1d
# # plt.rcParams['figure.figsize'] =(12,8)
# #
# # x = np.arange(2,10)
# # y = 1/(x)
# # # apply cubic spline interpolation
# # cs = CubicSpline(x, y, extrapolate=True)
# # # apply natural cubic spline interpolation
# # ns = CubicSpline(x, y,bc_type='natural', extrapolate=True)
# #
# # # Apply Linear interpolation
# # linear_int = interp1d(x,y)
# #
# # xs = np.arange(2, 9, 0.1)
# # ys = linear_int(xs)
# #
# # # plot linear interpolation
# # plt.plot(x, y,'o', label='data')
# # plt.plot(xs,ys, label="interpolation", color='green')
# # plt.legend(loc='upper right', ncol=2)
# # plt.title('Linear Interpolation')
# # plt.show()
# #
# # # define a new xs
# # xs = np.arange(1,15)
# # #plot cubic spline and natural cubic spline
# # plt.plot(x, y, 'o', label='data')
# # plt.plot(xs, 1/(xs), label='true')
# # plt.plot(xs, cs(xs), label="S")
# # plt.plot(xs, ns(xs), label="NS")
# # plt.plot(xs, ns(xs,2), label="NS''")
# # plt.plot(xs, ns(xs,1), label="NS'")
# #
# # plt.legend(loc='upper right', ncol=2)
# # plt.title('Cubic/Natural Cubic Spline Interpolation')
# # plt.show()
# #
# # # check for boundary condition
# # print("Value of double differentiation at boundary conditions are %s and %s"
# # 	%(ns(2,2),ns(10,2)))
# #
#
# # # Структура, описывающая сплайн на каждом сегменте сетки
# # class SplineTuple:
# #   def __init__(self, a, b, c, d, x):
# #     self.a = a
# #     self.b = b
# #     self.c = c
# #     self.d = d
# #     self.x = x
# #
# #
# # # Построение сплайна
# # # x - узлы сетки, должны быть упорядочены по возрастанию, кратные узлы запрещены
# # # y - значения функции в узлах сетки
# # # n - количество узлов сетки
# # def BuildSpline(x, y, n):
# #   # Инициализация массива сплайнов
# #   splines = [SplineTuple(1,2,3,5,7,8,11,14,15) for _ in range(0, n)]
# #   for i in range(0, n):
# #     splines[i].x = x[i]
# #     splines[i].a = y[i]
# #
# #   splines[0].c = splines[n - 1].c = 0.0
# #
# #   # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц
# #   # Вычисление прогоночных коэффициентов - прямой ход метода прогонки
# #   alpha = [0.0 for _ in range(0, n - 1)]
# #   beta = [0.0 for _ in range(0, n - 1)]
# #
# #   for i in range(1, n - 1):
# #     hi = x[i] - x[i - 1]
# #     hi1 = x[i + 1] - x[i]
# #     A = hi
# #     C = 2.0 * (hi + hi1)
# #     B = hi1
# #     F = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)
# #     z = (A * alpha[i - 1] + C)
# #     alpha[i] = -B / z
# #     beta[i] = (F - A * beta[i - 1]) / z
# #
# #   # Нахождение решения - обратный ход метода прогонки
# #   for i in range(n - 2, 0, -1):
# #     splines[i].c = alpha[i] * splines[i + 1].c + beta[i]
# #
# #   # По известным коэффициентам c[i] находим значения b[i] и d[i]
# #   for i in range(n - 1, 0, -1):
# #     hi = x[i] - x[i - 1]
# #     splines[i].d = (splines[i].c - splines[i - 1].c) / hi
# #     splines[i].b = hi * (2.0 * splines[i].c + splines[i - 1].c) / 6.0 + (y[i] - y[i - 1]) / hi
# #   return splines
# #
# #
# # import matplotlib.pyplot as plt
# #
# #
# # # Структура, описывающая сплайн на каждом сегменте сетки
# # class SplineTuple:
# #   def __init__(self, a, b, c, d, x):
# #     self.a = a
# #     self.b = b
# #     self.c = c
# #     self.d = d
# #     self.x = x
# #
# #
# # # Построение сплайна
# # # x - узлы сетки, должны быть упорядочены по возрастанию, кратные узлы запрещены
# # # y - значения функции в узлах сетки
# # # n - количество узлов сетки
# # def BuildSpline(x, y, n):
# #   # Инициализация массива сплайнов
# #   splines = [SplineTuple(1,2,3,5,7,8,11,14,15)  for _ in range(0, n)]
# #   for i in range(0, n):
# #     splines[i].x = x[i]
# #     splines[i].a = y[i]
# #
# #   splines[0].c = splines[n - 1].c = 0.0
# #
# #   # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц
# #   # Вычисление прогоночных коэффициентов - прямой ход метода прогонки
# #   alpha = [0.0 for _ in range(0, n - 1)]
# #   beta = [0.0 for _ in range(0, n - 1)]
# #
# #   for i in range(1, n - 1):
# #     hi = x[i] - x[i - 1]
# #     hi1 = x[i + 1] - x[i]
# #     A = hi
# #     C = 2.0 * (hi + hi1)
# #     B = hi1
# #     F = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)
# #     z = (A * alpha[i - 1] + C)
# #     alpha[i] = -B / z
# #     beta[i] = (F - A * beta[i - 1]) / z
# #
# #   # Нахождение решения - обратный ход метода прогонки
# #   for i in range(n - 2, 0, -1):
# #     splines[i].c = alpha[i] * splines[i + 1].c + beta[i]
# #
# #   # По известным коэффициентам c[i] находим значения b[i] и d[i]
# #   for i in range(n - 1, 0, -1):
# #     hi = x[i] - x[i - 1]
# #     splines[i].d = (splines[i].c - splines[i - 1].c) / hi
# #     splines[i].b = hi * (2.0 * splines[i].c + splines[i - 1].c) / 6.0 + (y[i] - y[i - 1]) / hi
# #   return splines
# #
# #
# # # Вычисление значения интерполированной функции в произвольной точке
# # def Interpolate(splines, x):
# #   if not splines:
# #     return None  # Если сплайны ещё не построены - возвращаем NaN
# #
# #   n = len(splines)
# #   s = SplineTuple(0, 0, 0, 0, 0)
# #
# #   if x <= splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
# #     s = splines[0]
# #   elif x >= splines[n - 1].x:  # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
# #     s = splines[n - 1]
# #   else:  # Иначе x лежит между граничными точками сетки - производим бинарный поиск нужного эл-та массива
# #     i = 0
# #     j = n - 1
# #     while i + 1 < j:
# #       k = i + (j - i) // 2
# #       if x <= splines[k].x:
# #         j = k
# #       else:
# #         i = k
# #     s = splines[j]
# #
# #   dx = x - s.x
# #   # Вычисляем значение сплайна в заданной точке по схеме Горнера (в принципе, "умный" компилятор применил бы схему Горнера сам, но ведь не все так умны, как кажутся)
# #   return s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx;
# #
# #
# # x = [1,2,3,5,7,8,11,14,15]
# # y = [-3,3,5,4,1,-5,6,3,2]
# # new_x = 5
# #
# # spline = BuildSpline(x, y, len(x))
# # plt.scatter(x, y)
# # plt.plot(x, y)
# # plt.scatter(new_x, Interpolate(spline, new_x))
# # plt.show()
# # Вычисление значения интерполированной функции в произвольной точке
# # def Interpolate(splines, x):
# #   if not splines:
# #     return None  # Если сплайны ещё не построены - возвращаем NaN
# #
# #   n = len(splines)
# #   s = SplineTuple(0, 0, 0, 0, 0)
# #
# #   if x <= splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
# #     s = splines[0]
# #   elif x >= splines[n - 1].x:  # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
# #     s = splines[n - 1]
# #   else:  # Иначе x лежит между граничными точками сетки - производим бинарный поиск нужного эл-та массива
# #     i = 0
# #     j = n - 1
# #     while i + 1 < j:
# #       k = i + (j - i) // 2
# #       if x <= splines[k].x:
# #         j = k
# #       else:
# #         i = k
# #     s = splines[j]
# #
# #   dx = x - s.x
# #   # Вычисляем значение сплайна в заданной точке по схеме Горнера (в принципе, "умный" компилятор применил бы схему Горнера сам, но ведь не все так умны, как кажутся)
# #   return s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx;
# #
# #
# # spline = BuildSpline([1, 3, 7, 9], [5, 6, 7, 8], 4)
# # print(Interpolate(spline, 5))
# # import numpy as np
# # from math import sqrt
# #
# #
# # def cubic_interp1d(x0, x, y):
# #   """
# #   Interpolate a 1-D function using cubic splines.
# #     x0 : a float or an 1d-array
# #     x : (N,) array_like
# #         A 1-D array of real/complex values.
# #     y : (N,) array_like
# #         A 1-D array of real values. The length of y along the
# #         interpolation axis must be equal to the length of x.
# #
# #   Implement a trick to generate at first step the cholesky matrice L of
# #   the tridiagonal matrice A (thus L is a bidiagonal matrice that
# #   can be solved in two distinct loops).
# #
# #   additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
# #   """
# #   x = np.asfarray(x)
# #   y = np.asfarray(y)
# #
# #   # remove non finite values
# #   # indexes = np.isfinite(x)
# #   # x = x[indexes]
# #   # y = y[indexes]
# #
# #   # check if sorted
# #   if np.any(np.diff(x) < 0):
# #     indexes = np.argsort(x)
# #     x = x[indexes]
# #     y = y[indexes]
# #
# #   size = len(x)
# #
# #   xdiff = np.diff(x)
# #   ydiff = np.diff(y)
# #
# #   # allocate buffer matrices
# #   Li = np.empty(size)
# #   Li_1 = np.empty(size - 1)
# #   z = np.empty(size)
# #
# #   # fill diagonals Li and Li-1 and solve [L][y] = [B]
# #   Li[0] = sqrt(2 * xdiff[0])
# #   Li_1[0] = 0.0
# #   B0 = 0.0  # natural boundary
# #   z[0] = B0 / Li[0]
# #
# #   for i in range(1, size - 1, 1):
# #     Li_1[i] = xdiff[i - 1] / Li[i - 1]
# #     Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
# #     Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
# #     z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]
# #
# #   i = size - 1
# #   Li_1[i - 1] = xdiff[-1] / Li[i - 1]
# #   Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
# #   Bi = 0.0  # natural boundary
# #   z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]
# #
# #   # solve [L.T][x] = [y]
# #   i = size - 1
# #   z[i] = z[i] / Li[i]
# #   for i in range(size - 2, -1, -1):
# #     z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]
# #
# #   # find index
# #   index = x.searchsorted(x0)
# #   np.clip(index, 1, size - 1, index)
# #
# #   xi1, xi0 = x[index], x[index - 1]
# #   yi1, yi0 = y[index], y[index - 1]
# #   zi1, zi0 = z[index], z[index - 1]
# #   hi1 = xi1 - xi0
# #
# #   # calculate cubic
# #   f0 = zi0 / (6 * hi1) * (xi1 - x0) ** 3 + \
# #        zi1 / (6 * hi1) * (x0 - xi0) ** 3 + \
# #        (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0) + \
# #        (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
# #   return f0
# #
# #
# # if __name__ == '__main__':
# #   import matplotlib.pyplot as plt
# #
# #   x = np.linspace(0, 10, 11)
# #   y = np.sin(x)
# #   plt.scatter(x, y)
# #
# #   x_new = np.linspace(0, 10, 201)
# #   plt.plot(x_new, cubic_interp1d(x_new, x, y))
# #
# #   plt.show()
#
# # def newton_interpolation(x_values, y_values, x):
# #   """
# #   :param x_values: список значений x
# #   :param y_values: список значений y
# #   :param x: значение, для которого нужно найти приближенное значение y
# #   :return: приближенное значение y, рассчитанное с помощью интерполяционного полинома Ньютона
# #   """
# #   n = len(x_values)
# #   # Инициализация разделенных разностей
# #   f = [[None] * n for _ in range(n)]
# #   for i in range(n):
# #     f[i][0] = y_values[i]
# #   # Вычисление разделенных разностей
# #   for j in range(1, n):
# #     for i in range(n - j):
# #       f[i][j] = (f[i + 1][j - 1] - f[i][j - 1]) / (x_values[i + j] - x_values[i])
# #   # Вывод таблицы конечных разностей
# #   print("Таблица конечных разностей:")
# #   for row in f:
# #     for elem in row:
# #       if elem is not None:
# #         print("{:.4f}".format(elem), end="\t")
# #       else:
# #         print("\t", end="")
# #     print()
# #   # Вычисление значения интерполяционного полинома
# #   y = 0
# #   for i in range(n):
# #     prod = f[0][i]
# #     for j in range(i):
# #       prod *= (x - x_values[j])
# #     y += prod
# #   return y
# #
# #
# # x_values = [1, 2, 3, 5, 7, 8, 11, 14, 15]
# # y_values = [-3, 3, 5, 4, 1, -5, 6, 3, 2]
# # x = 1
# # y = newton_interpolation(x_values, y_values, x)
# # print("Значение интерполяционного полинома в точке x = {:.2f}: {:.4f}".format(x, y))
# # # import numpy as np
# # import sympy as sp
# # x = sp.symbols("x")
# # x_numbers = np.array([1, 2, 3, 5, 7, 8, 11, 14, 15])
# # y_numbers = np.array([-3, 3, 5, 4, 1, -5, 6, 3, 2])
# # n = len(x_numbers)
# # # import numpy as np
# # from sympy import *
# # import matplotlib.pyplot as plt
# #
# #
# # def f(x):
# #     return 1 / (1 + x ** 2)
# #
# #
# # def cal(begin, end, i):
# #     by = f(begin)
# #     ey = f(end)
# #     I = Ms[i] * ((end - n) ** 3) / 6 + Ms[i + 1] * ((n - begin) ** 3) / 6 + (by - Ms[i] / 6) * (end - n) + (
# #             ey - Ms[i + 1] / 6) * (n - begin)
# #     return I
# #
# #
# # def ff(x):  # f[x0, x1, ..., xk]
# #     ans = 0
# #     for i in range(len(x)):
# #         temp = 1
# #         for j in range(len(x)):
# #             if i != j:
# #                 temp *= (x[i] - x[j])
# #         ans += f(x[i]) / temp
# #     return ans
# #
# #
# # def calM():
# #     lam = [1] + [1 / 2] * 9
# #     miu = [1 / 2] * 9 + [1]
# #     # Y = 1 / (1 + n ** 2)
# #     # df = diff(Y, n)
# #     x = np.array(range(11)) - 5
# #     # ds = [6 * (ff(x[0:2]) - df.subs(n, x[0]))]
# #     ds = [6 * (ff(x[0:2]) - 1)]
# #     for i in range(9):
# #         ds.append(6 * ff(x[i: i + 3]))
# #     # ds.append(6 * (df.subs(n, x[10]) - ff(x[-2:])))
# #     ds.append(6 * (1 - ff(x[-2:])))
# #     Mat = np.eye(11, 11) * 2
# #     for i in range(11):
# #         if i == 0:
# #             Mat[i][1] = lam[i]
# #         elif i == 10:
# #             Mat[i][9] = miu[i - 1]
# #         else:
# #             Mat[i][i - 1] = miu[i - 1]
# #             Mat[i][i + 1] = lam[i]
# #     ds = np.mat(ds)
# #     Mat = np.mat(Mat)
# #     Ms = ds * Mat.I
# #     return Ms.tolist()[0]
# #
# #
# # def calnf(x):
# #     nf = []
# #     for i in range(len(x) - 1):
# #         nf.append(cal(x[i], x[i + 1], i))
# #     return nf
# #
# #
# # def calf(f, x):
# #     y = []
# #     for i in x:
# #         y.append(f.subs(n, i))
# #     return y
# #
# #
# # def nfSub(x, nf):
# #     tempx = np.array(range(11)) - 5
# #     dx = []
# #     for i in range(10):
# #         labelx = []
# #         for j in range(len(x)):
# #             if x[j] >= tempx[i] and x[j] < tempx[i + 1]:
# #                 labelx.append(x[j])
# #             elif i == 9 and x[j] >= tempx[i] and x[j] <= tempx[i + 1]:
# #                 labelx.append(x[j])
# #         dx = dx + calf(nf[i], labelx)
# #     return np.array(dx)
#
# #
# # def draw(nf):
# #     plt.rcParams['font.sans-serif'] = ['SimHei']
# #     plt.rcParams['axes.unicode_minus'] = False
# #     x = np.linspace(-5, 5, 101)
# #     y = f(x)
# #     Ly = nfSub(x, nf)
# #     plt.plot(x, y, label='Примитивный')
# #     plt.plot(x, Ly, label='Функция интерполяции кубическим сплайном')
# #     plt.xlabel('x')
# #     plt.ylabel('y')
# #     plt.legend()
# #
# #     plt.savefig('1.png')
# #     plt.show()
# #
# #
# # def lossCal(nf):
# #     x = np.linspace(-5, 5, 101)
# #     y = f(x)
# #     Ly = nfSub(x, nf)
# #     Ly = np.array(Ly)
# #     temp = Ly - y
# #     temp = abs(temp)
# #     print(temp.mean())
# #
# #
# # if __name__ == '__main__':
# #     x = np.array(range(11)) - 5
# #     y = f(x)
# #
# #     n, m = symbols('n m')
# #     init_printing(use_unicode=True)
# #     Ms = calM()
# #     nf = calnf(x)
# #     draw(nf)
# #     lossCal(nf)
# # # Implementation of Linear Interpolation using Python3 code
# # # Importing library
# # from scipy.interpolate import interp1d
# #
# # X = [1, 2, 3, 5, 7, 8, 11, 14, 15]  # random x values
# # Y = [-3, 3, 5, 4, 1, -5, 6, 3, 2]  # random y values
# #
# # # test value
# # interpolate_x = 6
# # # Finding the interpolation
# # y_interp = interp1d(X, Y)
# # print("Value of Y at x = {} is".format(interpolate_x),
# #       y_interp(interpolate_x))
# # # Python3 code
# # # Implementing Linear interpolation
# # # Creating Function to calculate the
# # # linear interpolation
# #
# # def interpolation(d, x):
# #     output = d[0][1] + (x - d[0][0]) * ((d[1][1] - d[0][1]) / (d[1][0] - d[0][0]))
# #
# #     return output
# #
# #
# # # Driver Code
# # data = [[2019, 12124], [2021, 5700]]
# #
# # year_x = 2020
# #
# # # Finding the interpolation
# # print("Population on year {} is".format(year_x),
# #       interpolation(data, year_x))
# #
# #
# # w = sp.prod([x - i for i in x_numbers])
# # w
# # df_w = sp.diff(w)
# # L = sp.simplify(sum([y_numbers[i] * (w/((x-x_numbers[i])*df_w.subs({x: x_numbers[i]})))
# #                      for i in range(n)]))
# # L
# # [L.subs({x:x_numbers[i]}) for i in range(n)]
# #
# # # Python program to approximate
# # # the cube root of 27
# # guess = 0.0#догадываться
# # cube = 27
# # increment = 0.0001#ghbhjcn показателей
# # epsilon = 0.1
# #
# # # Finding the approximate value
# # while abs(guess ** 3 - cube) >= epsilon:
# #   guess += increment
#
# # Checking the approximate value
# # if abs(guess ** 3 - cube) >= epsilon:
# #   print("Failed on the cube root of", cube)
# # else:
# #   print(guess, "is close to the cube root of", cube)
#
# # from scipy import interpolate
# # import numpy as np
# #
# # fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 16), layout="tight")
# # for ax, N in zip(axs.flatten(), n_points):
# #     x_data, y_data = generate_data(f, L, R, N)
# #     plot_problem(ax, x_data, y_data)
# #
# #     poly = interpolate.KroghInterpolator(x_data, y_data)
# #     y_poly = poly(xh)
# #     ax.plot(xh, y_poly, label="Интерполирующий полином")
# #
# #     ax.set_title(f"Результат интерполяции ${N=}$")
# #     ax.legend()
# #
# #
# # L = 0
# # R = np.pi
# # n_points = (4, 10, 64, 65)
# # f = np.sin
# #
# # xh = np.linspace(L, R, 100)
# # yh = f(xh)
# #
# #
# # def generate_data(f, L, R, n):
# #     x_data = np.linspace(L, R, n - 1)
# #     y_data = f(x_data)
# #     return x_data, y_data
# #
# #
# # def plot_problem(ax, x_data, y_data):
# #     ax.plot(xh, yh, linewidth=0.5, label="Исходная зависимость")
# #     ax.scatter(x_data, y_data, marker="x", color="red", label="Известные данные")
# #     ax.set_xlabel("$x$")
# #     ax.set_ylabel("$y$")
# #
# #
# # fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 16), layout="tight")
# # for ax, N in zip(axs.flatten(), n_points):
# #         x_data, y_data = generate_data(f, L, R, N)
# #         plot_problem(ax, x_data, y_data)
# #         ax.set_title(f"Исходные данные ${N=}$")
# #         ax.legend()

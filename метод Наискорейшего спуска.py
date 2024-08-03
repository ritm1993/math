# import random
# import numpy as np
# import matplotlib.pyplot as plt
#
# """
# Метод наискорейшего спуска
#  Функция Розенброка
#  Функция f (x) = 100 * (x (2) -x (1). ^ 2). ^ 2+ (1-x (1)). ^ 2
#  Градиент g (x) = (- 400 * (x (2) -x (1) ^ 2) * x (1) -2 * (1-x (1)), 200 * (x (2) -x ( 1) ^ 2)) ^ (Т)
# """
#
#
# def goldsteinsearch(f, df, d, x, alpham, rho, t):
#   '''
#        Функция линейного поиска
#        Число f, производная df, текущая точка итерации x и текущее направление поиска d
#   '''
#   flag = 0
#
#   a = 0
#   b = alpham
#   fk = f(x)
#   gk = df(x)
#
#   phi0 = fk
#   dphi0 = np.dot(gk, d)
#   # print(dphi0)
#   alpha = b * random.uniform(0, 1)
#
#   while (flag == 0):
#     newfk = f(x + alpha * d)
#     phi = newfk
#     # print(phi,phi0,rho,alpha ,dphi0)
#     if (phi - phi0) <= (rho * alpha * dphi0):
#       if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
#         flag = 1
#       else:
#         a = alpha
#         b = b
#         if (b < alpham):
#           alpha = (a + b) / 2
#         else:
#           alpha = t * alpha
#     else:
#       a = a
#       b = alpha
#       alpha = (a + b) / 2
#   return alpha
#
#
# def rosenbrock(x):
#   # Функция: f (x) = 100 * (x (2) -x (1). ^ 2). ^ 2 + (1-x (1)). ^ 2
#   return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
#
#
# def jacobian(x):
#   # Градиент g (x) = (-400 * (x (2) -x (1) ^ 2) * x (1) -2 * (1-x (1)), 200 * (x (2) -x (1) ^ 2)) ^ (Т)
#   return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
#
#
# def steepest(x0):
#   print('Начальная точка:')
#
#
# print('x0', '\n')
# imax = 20000
# W = np.zeros((2, imax))
# epo = np.zeros((2, imax))
# W[:, 0] = x0
# i = 1
# x = x0
# grad = jacobian(x)
# delta = sum(grad ** 2)  # начальная ошибка
#
# f = open("Самый быстрый.txt", 'w')
#
# while i < imax and delta > 10 ** (-5):
#   p = -jacobian(x)
#   x0 = x
#   alpha = goldsteinsearch(rosenbrock, jacobian, p, x, 1, 0.1, 2)
#   x = x + alpha * p
#   W[:, i] = x
#   if i % 5 == 0:
#     epo[:, i] = np.array((i, delta))
#     f.write(str(i) + "        " + str(delta) + "\n")
#     print(i, np.array((i, delta)))
#   grad = jacobian(x)
#   delta = sum(grad ** 2)
#   i = i + 1
#
#   print("Количество итераций:", i)
#   print("Приблизительное оптимальное решение:")
# print(x, '\n')
# W = W[:, 0: i]  # Запись точки итерации
#
# return [W, epo]
#
# if __name__ == "__main__":
#   X1 = np.arange(-1.5, 1.5 + 0.05, 0.05)
#   X2 = np.arange(-3.5, 4 + 0.05, 0.05)
#   [x1, x2] = np.meshgrid(X1, X2)
#   f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2  # заданная функция
#   plt.contour(x1, x2, f, 20)  # рисуем 20 контурных линий функции
#
# x0 = np.array([-1.2, 1])
# list_out = steepest(x0)
# W = list_out[0]
#
# epo = list_out[1]
#
# plt.plot(W[0, :], W[1, :], 'g * -')  # Рисуем траекторию схождения точки итерации
#
# plt.show()
import numpy as np
import matplotlib.pyplot as plot


radius = 8                                  # working plane radius
global_epsilon = 0.000000001                # argument increment for derivative
centre = (global_epsilon, global_epsilon)   # centre of the working circle
arr_shape = 100                             # number of points processed / 360
step = radius / arr_shape                   # step between two points


def differentiable_function(x, y):
    return np.sin(x) * np.exp((1 - np.cos(y)) ** 2) + \
           np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2


def rotate_vector(length, a):
    return length * np.cos(a), length * np.sin(a)


def derivative_x(epsilon, arg):
    return (differentiable_function(global_epsilon + epsilon, arg) -
            differentiable_function(epsilon, arg)) / global_epsilon


def derivative_y(epsilon, arg):
    return (differentiable_function(arg, epsilon + global_epsilon) -
            differentiable_function(arg, epsilon)) / global_epsilon


def calculate_flip_points():
    flip_points = np.array([0, 0])
    points = np.zeros((360, arr_shape), dtype=bool)
    cx, cy = centre

    for i in range(arr_shape):
        for alpha in range(360):
            x, y = rotate_vector(step, alpha)
            x = x * i + cx
            y = y * i + cy
            points[alpha][i] = derivative_x(x, y) + derivative_y(y, x) > 0
            if not points[alpha][i - 1] and points[alpha][i]:
                flip_points = np.vstack((flip_points, np.array([alpha, i - 1])))

    return flip_points


def pick_estimates(positions):
    vx, vy = rotate_vector(step, positions[1][0])
    cx, cy = centre
    best_x, best_y = cx + vx * positions[1][1], cy + vy * positions[1][1]

    for index in range(2, len(positions)):
        vx, vy = rotate_vector(step, positions[index][0])
        x, y = cx + vx * positions[index][1], cy + vy * positions[index][1]
        if differentiable_function(best_x, best_y) > differentiable_function(x, y):
            best_x = x
            best_y = y

    for index in range(360):
        vx, vy = rotate_vector(step, index)
        x, y = cx + vx * (arr_shape - 1), cy + vy * (arr_shape - 1)
        if differentiable_function(best_x, best_y) > differentiable_function(x, y):
            best_x = x
            best_y = y

    return best_x, best_y


def gradient_descent(best_estimates, is_x):
    derivative = derivative_x if is_x else derivative_y
    best_x, best_y = best_estimates
    descent_step = step
    value = derivative(best_y, best_x)

    while abs(value) > global_epsilon:
        descent_step *= 0.95
        best_y = best_y - descent_step \
            if derivative(best_y, best_x) > 0 else best_y + descent_step
        value = derivative(best_y, best_x)

    return best_y, best_x


def find_minimum():
    return gradient_descent(gradient_descent(pick_estimates(calculate_flip_points()), False), True)


def get_grid(grid_step):
    samples = np.arange(-radius, radius, grid_step)
    x, y = np.meshgrid(samples, samples)
    return x, y, differentiable_function(x, y)


def draw_chart(point, grid):
    point_x, point_y, point_z = point
    grid_x, grid_y, grid_z = grid
    plot.rcParams.update({
        'figure.figsize': (4, 4),
        'figure.dpi': 200,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4
    })
    ax = plot.figure().add_subplot(111, projection='3d')
    ax.scatter(point_x, point_y, point_z, color='red')
    ax.plot_surface(grid_x, grid_y, grid_z, rstride=5, cstride=5, alpha=0.7)
    plot.show()


if __name__ == '__main__':
    min_x, min_y = find_minimum()
    minimum = (min_x, min_y, differentiable_function(min_x, min_y))
    draw_chart(minimum, get_grid(0.05))

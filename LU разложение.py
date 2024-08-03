import numpy as np

A = np.array([[4, 2, -1],
              [-3, 1, 1],
              [-1, 1, 1]])

E1 = np.array([[1,  0, 0],
               [-4, 1, 0],
               [0,  0, 1]])

E2 = np.array([[1, 0, 0],
               [0, 1, 0],
               [2, 0, 1]])

E3 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 1, 1]])

E1_inverse = np.linalg.inv(E1)
E2_inverse = np.linalg.inv(E2)
E3_inverse = np.linalg.inv(E3)

U = E3.dot(E2).dot(E1).dot(A)
L = E1_inverse.dot(E2_inverse).dot(E3_inverse)

print("\nStep 1 &amp; 2: Upper traingular matrix of A using elementary matrices:")#Верхняя треугольная матрица A с использованием элементарных матриц
print(U)
print("\nStep 1 &amp; 3: Lower traingular matrix of A using inverse elementary matrices:")#Нижняя треугольная матрица A с использованием обратных элементарных матриц
print(L)

U_inverse = np.linalg.inv(U)
L_inverse = np.linalg.inv(L)

b1 = np.array([[-1],
               [-1],
               [-8]]) # column vector

c1 = L_inverse.dot(b1)
x1 = U_inverse.dot(c1)
print("\nStep 4a: Solve c1 given same left hand side matrix A but different right hand side b1:")#Решите c1, учитывая ту же левую часть матрицы A, но другую правую часть b1:
print(c1)
print("\nStep 5b: Solution x1 given same left hand side matrix A but different right hand side b1:")#Решение x1 дано с той же левой частью матрицы A, но с другой правой частью b1:
print(x1)

b2 = np.array([[28],
               [22],
               [-11]]) # column vector

c2 = L_inverse.dot(b2)
x2 = U_inverse.dot(c2)
print("\nStep 4a: Solve c2 given same left hand side matrix A but different right hand side b2:")#Решите c2, учитывая ту же левую часть матрицы A, но другую правую часть b2
print(c2)
print("\nStep 5b: Solution x2 given same left hand side matrix A but different right hand side b2:")#Решение x2 дано с той же левой частью матрицы A, но с другой правой частью b2:
print(x2)

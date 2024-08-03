import numpy

def Metod_Progonki(A,B):
    #известные константы
    k1 = -A[0,1]
    m1 = B[0]
    k2 = -A[A.shape[0] - 1, A.shape[1] - 2]
    m2 = B[B.shape[0] - 1]
    alfa = k1
    beta = m1
    #поиск альф и бет
    c = 2
    a = 0
    b = 1
    alf = [alfa]
    bet = [beta]
    for i in range(1, A.shape[0] - 1):
        beta = (B[i] - A[i,a] * beta) / (A[i,a] * alfa + A[i,b])
        alfa = -A[i,c] / (A[i,a] * alfa + A[i,b])
        a += 1
        b += 1
        c += 1
        alf.append(alfa)
        bet.append(beta)
    #расчет игриков
    y = (k2 * beta + m2) / (1 - k2 * alfa)
    otv = [y]
    for i in range(len(alf) - 1, -1, -1):
        y = alf[i] * y + bet[i]
        otv.append(y)
    #переворачиваем значения в списке
    otvet = []
    for i in reversed(otv):
        otvet.append(i)
    return otvet

#задаем матрицы
A = numpy.array([[1,1,0,0],[2,3,-1,0],[0,-1,-4,2],[0,0,-1,2]])
B = numpy.array([3, 7, -3, -2])

#вызов функции и вывод на экран
print(A)
print(B)
print(Metod_Progonki(A,B))

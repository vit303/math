import numpy as np

def gauss_no_pivoting(A, b):
    n = len(b)
    # Прямой ход
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
    
    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]
    
    return x

A = np.array([[5, 2.833333, 1],
              [3, 1.7, 7],
              [1, 8, 1]], dtype=np.float32)

b = np.array([11.666666, 13.4, 18], dtype=np.float32)

# Решение системы
solution_no_pivoting = gauss_no_pivoting(A.copy(), b.copy())
print("Решение без выбора главного элемента:", solution_no_pivoting)

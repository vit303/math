import numpy as np

def gauss_with_pivoting(A, b):
    n = len(b)
    # Прямой ход
    for i in range(n):
        # Выбор главного элемента
        max_row_index = np.argmax(np.abs(A[i:n, i])) + i
        A[[i, max_row_index]] = A[[max_row_index, i]]
        b[[i, max_row_index]] = b[[max_row_index, i]]
        
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
    
    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]
    
    return x

A = np.array([[5, 3, 1],
              [3, 1.79999, 7],
              [1, 8, 1]], dtype=np.float32)

b = np.array([12, 13.59998, 18], dtype=np.float32)

# Решение системы с выбором главного элемента
solution_with_pivoting = gauss_with_pivoting(A.copy(), b.copy())
print("Решение с выбором главного элемента:", solution_with_pivoting)

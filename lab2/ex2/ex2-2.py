import numpy as np

def is_diagonally_dominant(A):
    """ Проверка диагонального преобладания """
    for i in range(len(A)):
        if abs(A[i, i]) < sum(abs(A[i, :])) - abs(A[i, i]):
            return False
    return True

def transform_to_diagonally_dominant(A, b):
    """ Преобразование матрицы к диагонально преобладающей форме (если возможно) """
    n = len(A)
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r, i]))
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]
    return A, b

def simple_iteration_method(A, b, tol=1e-3, max_iterations=1000):
    """ Метод простых итераций """
    n = len(A)
    x = np.zeros(n)
    B = np.zeros_like(A)
    c = np.zeros(n)
    
    for i in range(n):
        c[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                B[i, j] = -A[i, j] / A[i, i]
    
    for _ in range(max_iterations):
        x_new = np.dot(B, x) + c
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Выбор матрицы (выберите нужную):
A = np.array([[2, 15, -1],
              [12, 2, 3],
              [1, 2, 16]], dtype=np.float64)

b = np.array([29, 25, 53], dtype=np.float64)

# Преобразуем, если необходимо
if not is_diagonally_dominant(A):
    A, b = transform_to_diagonally_dominant(A, b)

# Решение методом простых итераций
solution = simple_iteration_method(A, b, tol=1e-3)
print("Решение:", solution)

for precision in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
    solution = simple_iteration_method(A, b, tol=precision)
    print(f"Решение при точности {precision}: {solution}")
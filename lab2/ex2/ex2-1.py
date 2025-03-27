import numpy as np

def manual_zeros(shape):
    if isinstance(shape, int):
        return [0.0] * shape
    elif isinstance(shape, tuple) and len(shape) == 2:
        return [[0.0] * shape[1] for _ in range(shape[0])]
    else:
        raise ValueError("Неподдерживаемая форма массива")

def rearrange_rows(A, b):
    """Переставляет строки матрицы для обеспечения диагонального преобладания"""
    n = len(A)
    indices = list(range(n))
    indices.sort(key=lambda i: abs(A[i][i]) - sum(abs(A[i][j]) for j in range(n) if j != i), reverse=True)
    A = A[indices]
    b = b[indices]
    return A, b

def iterative_method(A, b, tol, max_iter=1000):
    """Решает СЛАУ методом простых итераций с заданной точностью tol"""
    n = len(A)
    x_old = manual_zeros(n)
    x_new = manual_zeros(n)
    for iteration in range(max_iter):
        for i in range(n):
            sum1 = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum1) / A[i][i]
        
        # Проверка критерия остановки (разность норм)
        if max(abs(x_new[i] - x_old[i]) for i in range(n)) < tol:
            break
        x_old = x_new.copy()
    return x_new

# Исходные данные
A = np.array([[5, 3, 1],
              [3, 1.79999, 7],
              [1, 8, 1]], dtype=np.float64)

b = np.array([12, 13.59998, 18], dtype=np.float64)

# Перестановка строк для обеспечения сходимости
A, b = rearrange_rows(A, b)

# Запуск расчётов с разной точностью
for precision in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
    solution = iterative_method(A, b, tol=precision)
    print(f"Решение при точности {precision}: {solution}")
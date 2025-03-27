import numpy as np

# Определяем матрицу A и вектор b с повышенной точностью (float64)
A = np.array([[5, 2.833333, 1],
              [3, 1.7, 7],
              [1, 8, 1]], dtype=np.float64)  

b = np.array([11.666666, 13.4, 18], dtype=np.float64) 

max_iter = 10000  # Увеличим максимальное число итераций

def null_approach(A, b, n):
    """Начальное приближение - решение по диагонали"""
    return [b[i] / A[i][i] for i in range(n)]

def iteration_process(A, b, max_iter, tol=1e-3):
    n = len(A)
    x_prev = null_approach(A, b, n)
    x_next = [0] * n
    
    for iteration in range(max_iter):
        for i in range(n):
            sum_ax = 0
            for j in range(n):
                if j != i:
                    sum_ax += A[i][j] * x_prev[j]
            x_next[i] = (b[i] - sum_ax) / A[i][i]
        
        # Проверка условия остановки - норма разности векторов
        error = max(abs(x_next[i] - x_prev[i]) for i in range(n))
        if error < tol:
            print(f"Сошлось за {iteration} итераций")
            break
            
        x_prev = x_next.copy()
    else:
        raise ValueError(f"Метод не сошёлся за {max_iter} итераций")
    
    return x_next

# Преобразованная система (диагональное преобладание)
A_modified = np.array([[5, 2.833333, 1],
                       [1, 8, 1],
                       [3, 1.7, 7]], dtype=np.float64)
b_modified = np.array([11.666666, 18, 13.4], dtype=np.float64)

print("\nРешение преобразованной системы с точностью 1e-15:")
try:
    res_modified = iteration_process(A_modified, b_modified, max_iter, tol=1e-15)
    print(res_modified)
except Exception as e:
    print(f"Ошибка: {e}")

# Проверка диагонального преобладания
def check_diagonal_dominance(A):
    n = len(A)
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if abs(A[i][i]) <= row_sum:
            return False
    return True

print("\nПроверка диагонального преобладания:")
print("Исходная матрица:", "удовлетворяет" if check_diagonal_dominance(A) else "не удовлетворяет")
print("Модифицированная матрица:", "удовлетворяет" if check_diagonal_dominance(A_modified) else "не удовлетворяет")

# Эксперимент с высокой точностью
print("\nЭксперимент с разной точностью (до 1e-15):")
tolerances = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
for tol in tolerances:
    try:
        res = iteration_process(A_modified, b_modified, max_iter, tol)
        print(f"Точность {tol:.0e}: Решение {res}")
    except Exception as e:
        print(f"Точность {tol:.0e}: Ошибка - {e}")

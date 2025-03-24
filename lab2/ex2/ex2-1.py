import numpy as np

# Определяем матрицу A и вектор b
A = np.array([[5, 2.833333, 1],
              [3, 1.7, 7],
              [1, 8, 1]], dtype=np.float32)

b = np.array([11.666666, 13.4, 18], dtype=np.float32)
# Максимальное количество итераций и точность
max_iter = 1000
tolerance = 1e-3

# Начальная точка
x = np.zeros(3)

# Итерационный процесс
for iteration in range(max_iter):
    x_new = np.zeros_like(x)
    
    # Вычисляем новое значение для каждого компонента
    x_new[0] = (b[0] - A[0][1]*x[1] - A[0][2]*x[2]) / A[0][0]
    x_new[1] = (b[1] - A[1][0]*x[0] - A[1][2]*x[2]) / A[1][1]
    x_new[2] = (b[2] - A[2][0]*x[0] - A[2][1]*x[1]) / A[2][2]

    # Проверка на сходимость
    if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
        break
        
    x = x_new

# Результаты
print(f'Решение: {x}, Количество итераций: {iteration + 1}')

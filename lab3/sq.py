import numpy as np
import matplotlib.pyplot as plt

# Пример данных
x = np.array([-2, 0, 1, 3, 4, 5])
y = np.array([7, 9, 6, 8, 9, 10])

# Степень полинома
degree = 25

# Создание матрицы A для полинома
A = np.vander(x, degree + 1)

# Решение системы уравнений для нахождения коэффициентов
coefficients = np.linalg.lstsq(A, y, rcond=None)[0]

# Генерация значений полинома
x_fit = np.linspace(min(x), max(x), 100)
y_fit = np.polyval(coefficients[::-1], x_fit)  # Порядок коэффициентов

# Визуализация
plt.scatter(x, y, color='black', label='Исходные данные')
plt.plot(x_fit, y_fit, color='magenta', label='МНК полинома')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Аппроксимация по коэффициентам (МНК)')
plt.grid()
plt.show()

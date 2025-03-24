import numpy as np

class TridiagonalMatrixSolver:
    def __init__(self, lower_diag, main_diag, upper_diag, rhs):
        self.lower_diag = lower_diag
        self.main_diag = main_diag
        self.upper_diag = upper_diag
        self.rhs = rhs

    def solve(self):
        n = len(self.main_diag)
        # Создаем массивы для хранения промежуточных значений
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)

        # Прямой проход
        c_prime[0] = self.upper_diag[0] / self.main_diag[0]
        d_prime[0] = self.rhs[0] / self.main_diag[0]

        for i in range(1, n):
            denominator = self.main_diag[i] - self.lower_diag[i-1] * c_prime[i-1]
            c_prime[i-1] = self.upper_diag[i-1] / denominator
            d_prime[i] = (self.rhs[i] - self.lower_diag[i-1] * d_prime[i-1]) / denominator

        # Обратный проход
        x = np.zeros(n)
        x[-1] = d_prime[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

# Пример использования
if __name__ == "__main__":
    # Пример трехдиагональной матрицы
    lower_diag = [1, 1]  # a[i] (нижняя диагональ)
    main_diag = [4, 4, 4]  # b[i] (главная диагональ)
    upper_diag = [1, 1]  # c[i] (верхняя диагональ)
    rhs = [5, 5, 5]  # правая часть

    solver = TridiagonalMatrixSolver(lower_diag, main_diag, upper_diag, rhs)
    solution = solver.solve()

    print("Решение системы:", solution)

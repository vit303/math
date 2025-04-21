import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

class Approx:
    def __init__(self):
        self.x = np.array([-2, 0, 1, 3, 5])
        self.y = np.array([7, 6, 10, 9, 10])
        self.n = len(self.x)
        
        self.fig, (self.ax, self.ax_table) = plt.subplots(2, 1, 
                                                         figsize=(12, 10),
                                                         gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(right=0.8, hspace=0.5)
        
        self.textboxes = []
        for i in range(5):
            ax_box_x = plt.axes([0.1 + i*0.15, 0.05, 0.1, 0.04])
            box_x = TextBox(ax_box_x, f'x{i+1}', initial=str(self.x[i]))
            box_x.on_submit(self.update_data)

            ax_box_y = plt.axes([0.1 + i*0.15, 0.01, 0.1, 0.04])
            box_y = TextBox(ax_box_y, f'y{i+1}', initial=str(self.y[i]))
            box_y.on_submit(self.update_data)
            
            self.textboxes.append((box_x, box_y))
        
        self.additional_textboxes = []
        for i in range(4):
            ax_box_x = plt.axes([0.82, 0.05 + i*0.05, 0.15, 0.04])
            box_x = TextBox(ax_box_x, f'Коэффициент {4 - i}', initial='0')
            self.additional_textboxes.append(box_x)

        self.ax_plot = plt.axes([0.82, 0.40, 0.15, 0.05])
        self.btn_plot = Button(self.ax_plot, 'Построить график')
        self.btn_plot.on_clicked(self.plot_function)

        self.ax_newton = plt.axes([0.82, 0.70, 0.15, 0.05])
        self.ax_lagrange = plt.axes([0.82, 0.65, 0.15, 0.05])
        self.ax_coef = plt.axes([0.82, 0.60, 0.15, 0.05])
        
        self.btn_newton = Button(self.ax_newton, 'Ньютона')
        self.btn_lagrange = Button(self.ax_lagrange, 'Лагранжа')
        self.btn_coef = Button(self.ax_coef, 'МНК (1-3 степени)')
        
        self.btn_newton.on_clicked(self.plot_newton)
        self.btn_lagrange.on_clicked(self.plot_lagrange)
        self.btn_coef.on_clicked(self.plot_coefficients)
        
        self.plot_initial()
        self.update_table()
    
    def update_data(self, text):
        try:
            for i, (box_x, box_y) in enumerate(self.textboxes):
                if i < len(self.x):
                    self.x[i] = float(box_x.text)
                    self.y[i] = float(box_y.text)
            
            self.n = len(self.x)
            self.plot_initial()
            self.update_table()
            plt.draw()
        except ValueError:
            print("Ошибка ввода данных. Пожалуйста, введите числа.")
    
    def plot_function(self, event):
        try:
            coeffs = [float(box.text) for box in self.additional_textboxes]
            x_plot = np.linspace(min(self.x) - 1, max(self.x) + 1, 1000)
            y_plot = self.graf(x_plot, coeffs)

            self.ax.plot(x_plot, y_plot, 'g-', label=f'Функция: {coeffs[0]:.2f}x⁴ + {coeffs[1]:.2f}x³ + {coeffs[2]:.2f}x² + {coeffs[3]:.2f}x')
            self.ax.legend()
            plt.draw()
        except ValueError:
            print("Ошибка ввода данных для коэффициентов. Пожалуйста, введите числа.")

    def graf(self, x, coeffs):
        return coeffs[0] * x**4 + coeffs[1] * x**3 + coeffs[2] * x**2 + coeffs[3] * x

    def update_table(self):
        self.ax_table.clear()
        self.ax_table.axis('off')
        
        col_labels = [f'x{i+1}' for i in range(len(self.x))] + [f'y{i+1}' for i in range(len(self.y))]
        cell_text = [list(map(str, self.x)) + list(map(str, self.y))]
        
        table = self.ax_table.table(cellText=cell_text,
                                     colLabels=col_labels,
                                     loc='center',
                                     cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    
    def lagrange(self, x_point):
        result = 0.0
        for i in range(self.n):
            term = self.y[i]
            for j in range(self.n):
                if i != j:
                    term *= (x_point - self.x[j]) / (self.x[i] - self.x[j])
            result += term
        return result
    
    class NewtonInterpolator:
        def __init__(self, x, y):
            self.x = np.array(x, dtype=float)
            self.y = np.array(y, dtype=float)
            self.div_diff = None
            self._compute_divided_differences()
            
        def _compute_divided_differences(self):
            n = len(self.x)
            self.div_diff = self.y.copy()
            
            for j in range(1, n):
                for i in range(n-1, j-1, -1):
                    self.div_diff[i] = (self.div_diff[i] - self.div_diff[i-1]) / (self.x[i] - self.x[i-j])
        
        def evaluate(self, x_point):
            if self.div_diff is None:
                self._compute_divided_differences()
                
            n = len(self.x)
            result = self.div_diff[-1]
            
            for i in range(n-2, -1, -1):
                result = result * (x_point - self.x[i]) + self.div_diff[i]
                
            return result
    
    def newton(self, x_point):
        interpolator = self.NewtonInterpolator(self.x, self.y)
        return interpolator.evaluate(x_point)

    def least_squares(self, degree):
        A = np.vstack([self.x**d for d in range(degree, -1, -1)]).T
        coef = np.linalg.lstsq(A, self.y, rcond=None)[0]
        return coef
    
    def print_results(self, method):
        print(f"\nРезультаты для метода {method}:")
        print("X значения:", self.x)
        print("Y значения:", self.y)
        
        if method == "Ньютона":
            interpolator = self.NewtonInterpolator(self.x, self.y)
            print("\nРазделенные разности:", interpolator.div_diff)
        
        elif method == "Лагранжа":
            print("\nБазисные полиномы Лагранжа:")
            for i in range(self.n):
                print(f"L_{i}(x) = ", end="")
                terms = []
                for j in range(self.n):
                    if i != j:
                        terms.append(f"(x - {self.x[j]})/{self.x[i] - self.x[j]}")
                print(" * ".join(terms))
        
        elif method == "МНК":
            for degree in [1, 2, 3]:
                coef = self.least_squares(degree)
                print(f"\nКоэффициенты для степени {degree}:")
                print("y = ", end="")
                for i, c in enumerate(coef):
                    power = degree - i
                    if power > 1:
                        print(f"{c:.4f}x^{power} + ", end="")
                    elif power == 1:
                        print(f"{c:.4f}x + ", end="")
                    else:
                        print(f"{c:.4f}")
    
    def plot_initial(self):
        self.ax.clear()
        self.ax.scatter(self.x, self.y, color='black', s=100, label='Исходные данные', zorder=5)
        self.ax.set_title('Выберите метод аппроксимации')
        self.ax.grid(True)
        self.ax.legend()
    
    def plot_newton(self, event):
        interpolator = self.NewtonInterpolator(self.x, self.y)
        x_plot = np.linspace(min(self.x)-1, max(self.x)+1, 1000)
        y_newton = np.array([interpolator.evaluate(x) for x in x_plot])

        self.ax.clear()
        self.ax.scatter(self.x, self.y, color='black', s=100, label='Исходные данные', zorder=5)
        self.ax.plot(x_plot, y_newton, 'r-', linewidth=2, label='Интерполяция Ньютона')

        # Проверка точности в узлах интерполяции
        for xi, yi in zip(self.x, self.y):
            computed = interpolator.evaluate(xi)
            if not np.isclose(computed, yi, atol=1e-8):
                print(f"Предупреждение: P({xi}) = {computed} (ожидалось {yi})")
                self.ax.scatter([xi], [computed], color='orange', s=80, marker='x', 
                              zorder=6, label='Ошибка интерполяции' if xi == self.x[0] else "")

        self.ax.set_title('Аппроксимация методом Ньютона')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='best')
        self.print_results("Ньютона")
        plt.draw()
    
    def plot_lagrange(self, event):
        x_plot = np.linspace(min(self.x)-1, max(self.x)+1, 1000)
        y_lagrange = np.array([self.lagrange(x) for x in x_plot])
        
        self.ax.clear()
        self.ax.scatter(self.x, self.y, color='black', s=100, label='Исходные данные', zorder=5)
        self.ax.plot(x_plot, y_lagrange, 'b-', label='Интерполяция Лагранжа')
        self.ax.set_title('Аппроксимация методом Лагранжа')
        self.ax.grid(True)
        self.ax.legend()
        self.print_results("Лагранжа")
        plt.draw()
    
    def plot_coefficients(self, event):
        x_plot = np.linspace(min(self.x)-1, max(self.x)+1, 1000)
        
        coef1 = self.least_squares(1)
        coef2 = self.least_squares(2)
        coef3 = self.least_squares(3)
        
        y_ls1 = np.polyval(coef1, x_plot)
        y_ls2 = np.polyval(coef2, x_plot)
        y_ls3 = np.polyval(coef3, x_plot)
        
        self.ax.clear()
        self.ax.scatter(self.x, self.y, color='black', s=100, label='Исходные данные', zorder=5)
        self.ax.plot(x_plot, y_ls1, 'g-', label='МНК 1 степени')
        self.ax.plot(x_plot, y_ls2, 'm-', label='МНК 2 степени')
        self.ax.plot(x_plot, y_ls3, 'c-', label='МНК 3 степени')
        self.ax.set_title('Аппроксимация методом наименьших квадратов')
        self.ax.grid(True)
        self.ax.legend()
        plt.draw()

if __name__ == "__main__":
    app = Approx()
    plt.show()
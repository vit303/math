import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -x**4 + 15*x**2 + 12*x - 10

# массив значений x
x = np.linspace(-5, 5, 400) 
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = -x^4 + 15x^2 + 12x - 10', color='blue')
plt.axhline(0, color='black', lw=0.5, ls='--')  # Горизонтальная линия y=0
plt.axvline(0, color='black', lw=0.5, ls='--')  # Вертикальная линия x=0
plt.title('График уравнения -x^4 + 15x^2 + 12x - 10')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.ylim(-50, 50)  
plt.show()


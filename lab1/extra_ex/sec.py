def f(x):
    return 100*x*x - 10000*x + x**4 - 15*x**2 - 12*x + 10

def secant_method(x0, x1, tol=1e-3, max_iter=100):
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1) < tol:
            return x1, i 
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)  # Формула секущих
        x0, x1 = x1, x_new 
    return x1 , i

# Поиск корней методом секущих
roots_secant = [
    secant_method(0, 1),        
    secant_method(20, 25),   
]

print("Корни: ", roots_secant)

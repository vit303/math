def f(x):
    return -x**4 + 15*x**2 + 12*x - 10

def g(x, m):
    return x - (1/m)*(-x**4 + 15*x**2 + 12*x - 10)

def simple_iteration(x1, x2, m, tol=1e-3, max_iter=100):
    x0 = (x1+x2)/2
    for i in range(max_iter):
        x_res = g(x0, m)
        if abs(x_res - x0) < tol:
            return x_res, i
        x0 = x_res
    return x0, i 

# Поиск корней методом секущих
roots_secant = [
    simple_iteration(-4, -3, 79), 
    simple_iteration(-2, 0, -20),
    simple_iteration(0, 1, 27),
    simple_iteration(4, 5, -218),            
]

print("Корни: ", roots_secant)
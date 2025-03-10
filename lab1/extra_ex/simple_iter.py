def f(x):
    return 100*x*x - 10000*x + x**4 - 15*x**2 - 12*x + 10

def g(x, m):
    return x - (1/m)*(100*x*x - 10000*x + x*x*x*x - 15*x**2 - 12*x + 10)

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
    simple_iteration(0, 1, -9927),       
    #simple_iteration(20, 25, -6231),   Result too large
]

print("Корни: ", roots_secant)
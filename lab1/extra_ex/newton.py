def df(x):
    return 200*x - 10000 +4*x**3 - 30*x - 12  # Производная f(x)

def f(x):
    return 100*x*x - 10000*x + x**4 - 15*x**2 - 12*x + 10

def newton_method(x0, tol=1e-3, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            return x, i
        x -= fx / dfx
    return x, i

# Поиск корней методом Ньютона
initial_guesses = [0, 20]
roots_newton = [newton_method(x0) for x0 in initial_guesses]
print("Корни: ", roots_newton)

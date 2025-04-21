# Imports
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

plt.style.use("ggplot")

# ANSI Color for OUTPUT
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"  # Сброс цвета

"""
    Метод наименьших квадратов заключается в том, что оценка определяется из условия минимизации суммы квадратов отклонений выборочных данных   
от их теоретических значений.
"""

# Этап 1: Сбор (Ввод) данных
N = 100  # число экспериментов
sigma = 3  # стандартное отклонение наблюдаемых значений
a = 0.5  # теоретическое значение параметра a
b = 2  # теоретическое значение параметра b

x = np.array(range(N))  # Вспомогательный вектор: x = [1, 2, 3, 4, ..., N - 1]

f = np.array([a * z + b for z in range(N)])  # Вычисляем значение теоретической функции

y = f + np.random.normal(0, sigma,
                         N)  # Добавляем к ней случайные отклонения(ошибку) для моделирования результатов наблюдения

print(f"{CYAN}Этап 1: Сбор данных наблюдений (5 значений){RESET}\nx = {x[:5]}\ny = {y[:5]}\n")

# Набор точек y на данном этапе
plt.figure(figsize=(14, 7))
plt.scatter(x, y, c='red')
plt.title("Набор точек 'y' на начальном этапе")
plt.grid(True)
plt.show()

# Этап 2: Формулировка уравнения линейной регрессии
print(
    f"{CYAN}Этап 2: Формулировка уравнения линейной регрессии{RESET}\nУравнение регрессии: {YELLOW}y = bx + a{RESET}\nгде a - свободный член (интерцепт), b - коэффициент наклона\n")

# Этап 3: Вычисление коэффициентов линейной регрессии (Минимизция суммы квадратов отклонения)
print(f"{CYAN}Этап 3: Минимизация суммы квадратов отклонения (Вычисление коэффициентов){RESET}")
print(f"Формула:{YELLOW}")
# Определяем символы (Формула)
n_ = sp.symbols('n', integer=True, positive=True)
a_, b_ = sp.symbols('a b')
x_ = sp.IndexedBase('x')
y_ = sp.IndexedBase('y')
i_ = sp.symbols('i', integer=True)

# Формула суммы квадратов отклонений
S = sp.Sum((y_[i_] - (a_ + b_ * x_[i_])) ** 2, (i_, 1, n_))

# Выводим формулу в консоль
sp.pprint(S)

# Этап 4: Вычисление сумм
print(f"\n{CYAN}Этап 4: Вычисление сумм{RESET}")
n = len(x)
sum_x = np.sum(x) / N
sum_y = np.sum(y) / N
sum_x_squared = np.sum(x ** 2)
sum_xy = np.sum(x * y)

mx = x.sum() / N
my = y.sum() / N
a2 = np.dot(x.T, x) / N
a11 = np.dot(x.T, y) / N

kk = (a11 - mx * my) / (a2 - mx ** 2)
bb = my - kk * mx

ff = np.array([kk * z + bb for z in range(N)])

a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
b = (sum_y - a * sum_x) / n
print(
    f"Количество точек данных {YELLOW}n = {GREEN}{n}{RESET}\nСумма {YELLOW}x = {GREEN}{sum_x}{RESET}\nСумма {YELLOW}y = {GREEN}{sum_y}{RESET}\nСумма {YELLOW}x^2 = {GREEN}{sum_x_squared}{RESET}\nСумма {YELLOW}x*y = {GREEN}{sum_xy}{RESET}\nСвободный член {YELLOW}(a) = {GREEN}{a}{RESET}\nКоэффициент наклона {YELLOW}(b) = {GREEN}{b}{RESET}")

# Построение линии регрессии: y = ax + b
y_predict = a * x + b
print(f"{YELLOW}Линия Аппроксимации{RESET} = {GREEN}{y_predict[:5]}{RESET}")

# Графики
plt.figure(figsize=(14, 7))
plt.scatter(x, y, color='blue', label='Исходные данные')

ff = np.array([kk * z + bb for z in range(N)])
plt.plot(f, linewidth=2, color='green', label='Теоретическая функция')
plt.plot(ff, linewidth=2, color='red', label='Линия Аппроксимации')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


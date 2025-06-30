import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2+5*x+5

def f_grad(x):
    return 6*x+5


x = np.linspace(-10,10,1000)
y = f(x)




tol = 0.001


plt.plot(x, y)

x0 = np.random.randint(low=-10, high= 10)

y0 = f(x0)

plt.scatter(x0, y0, color = 'red')


for i in range(1000):



    x_grad = f_grad(x0)
    decey = 0.01 * x_grad
    if abs(decey) <= tol:
        print(f'itterations: {i}')
        break

    x0 = x0 - decey 
    y0 = f(x0)


    plt.scatter(x0, y0, c = 'green')


    

print(x0)
plt.xlabel('x')
plt.ylabel('y')


plt.grid()

plt.show()

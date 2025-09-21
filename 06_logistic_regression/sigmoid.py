import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def sigmoid(x1, x2):
    z = beta1 * x1 + beta2 * x2 + beta0
    return 1/(1+ np.exp(-z))

x = np.linspace(-10, 10, 1000)
y = sigmoid(x)


plt.plot(x, y)

plt.show()

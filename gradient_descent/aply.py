import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df  = pd.read_csv('advertising.csv')


X = df['TV'].values.reshape(-1, 1)
Y = df['Sales'].values.reshape(-1, 1)

X = scaler.fit_transform(X)
# Y = scaler.fit_transform(Y)


# print(X)
# print(Y)

#Hyperparametr


def loss(beta0, beta1):
    return np.sum((Y - (beta0 + beta1*X).ravel()) ** 2)**0.5 / X.shape[0]


def grad_loss(beta0, beta1):
    beta0_grad = -2*np.sum(((Y - (beta0 + beta1*X)).ravel()))
    beta1_grad = -2*np.sum(((Y - (beta0 + beta1*X)).ravel())*X.ravel())
    return beta0_grad, beta1_grad


beta0 = np.random.randn()
beta1 = np.random.randn()

print(beta0, beta1)

lr = 0.001
tol = 0.00001
max_iter = 10000

# beta0_grad, beta1_grad = grad_loss(beta0, beta1)

lossi = []
for i in range(max_iter):
    beta0_grad, beta1_grad = grad_loss(beta0, beta1)
    if ((lr*beta0_grad)**2 + (lr*beta1_grad)**2)**0.5 < tol:
        print(f'Convergence after {i}')
        break
    beta0 = beta0 - lr*beta0_grad
    beta1 = beta1 - lr*beta1_grad
    l = loss(beta0, beta1)
    lossi.append(l)

plt.plot(lossi[:10])
# plt.show()



print(beta0, beta1)


from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X, Y)

print(lin.intercept_)
print(lin.coef_)



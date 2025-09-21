import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

df  = pd.read_csv('advertising.csv')


X = df.drop(columns ='Sales').values
Y = df['Sales'].values

X = scaler.fit_transform(X)

class GradientDescent:
    def __init__(self, lr, max_iter, tol):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, Y):

        X = np.array(X)
        Y = np.array(Y)
        n = X.shape[0]

        X = np.c_[np.ones(n), X]

        n_features = X.shape[1]
        coef = np.random.randn(n_features)
        def grad_loss(coef):
            y_pred = X@coef
            e = Y - y_pred
            coef_grad = (-2/n)*X.T @ e
            return coef_grad
        
        for i in range(self.max_iter):
            coef_grad = grad_loss(coef)
            coef = coef - self.lr * coef_grad
            if np.linalg.norm(self.lr * coef_grad) < self.tol:
                break
        print(coef)
        self.intercept_ = coef[0]
        self.coef_ = coef[1:]
        return self
    
    def predict(self, X):
        n = X.shape[0]
        X = np.c_[np.ones(n), X]
        return X @ np.r_[self.intercept_, self.coef_]
    


gd = GradientDescent(lr = 0.01,max_iter =1000, tol = 0.001)
gd.fit(X,Y)
print(gd.coef_)
print(gd.intercept_)

print(gd.predict(X))


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X,Y)


print(lin_reg.coef_)
print(lin_reg.intercept_)
print(lin_reg.predict(X))



plt.scatter(Y, df['TV'], c ='blue')
plt.scatter(gd.predict(X), df['TV'], c= 'green')
plt.scatter(lin_reg.predict(X), df['TV'], c= 'red')

plt.show()

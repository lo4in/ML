import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = "advertising.csv"


df = pd.read_csv(path)

#print(df.head(10))

X = df[['TV', 'Radio', 'Newspaper']].values
Y = df['Sales'].values
N = X.shape[0]
X = np.c_[np.ones(N), X]

beta = np.linalg.inv(X.T@X)@ X.T@Y


y_p = X@beta

plt.plot(y_p, X)

#plt.show()


print(beta)



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()


lin_reg.fit(X, Y)

print(lin_reg.coef_)
print(lin_reg.intercept_)

print(lin_reg.score(X, Y))
print(df.corr())
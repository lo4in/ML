import numpy as np
import pandas as pd

path = "data/heart_diceases.csv"


df = pd.read_csv(path)
print(df.head())
X = df.drop(columns = 'target').values

Y = df['target'].values
print(Y)
from sklearn.preprocessing import StandardScaler

scal = StandardScaler()

X = scal.fit_transform(X)


ones = np.ones((X.shape[0], 1))

X = np.hstack((ones, X))
N = np.shape(X)
print(X)
print(N[1])
beta = np.zeros(N[1])
print(beta)

def sigmoid(X, beta):
    z = X * beta
    return 1/(1+np.exp(-z))


y_pred = sigmoid()





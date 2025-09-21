import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

path = "homework/housing-3.csv"

df = pd.read_csv(path)


df = df.replace({'yes': 0 , 'no': 1})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

df = df.replace({False: 0 , True: 1})


x_pre = df.drop(columns = ['price']).values
N = x_pre.shape[0]
x_pre = np.c_[np.ones(N), x_pre]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


Y = df_scaled['price'].values
X = df_scaled.drop(columns = ['price']).values

N = X.shape[0]
X = np.c_[np.ones(N), X]

beta = np.linalg.inv(X.T@X)@X.T@Y

print(beta)

example = X[0]

predict = np.dot(example, beta)

print(predict)

# print(df_scaled['price'])

y_p = X@ beta
print(y_p)


MAE = np.mean(np.abs(y_p - Y))
print("MAE: ", MAE)


MSE = np.mean((y_p - Y)**2)

print("MSE: ", MSE)


from sklearn.metrics import r2_score

r2 = r2_score(Y, y_p)
print("R^2:", r2)

print(df_scaled.head())


x = '1, 2400, 4, 3, 2, 1, 0, 1, 0, 1, 2, 1, 1, 0'
x = x.split(',')
x = [float(i.strip()) for i in x]
x = np.array([x])

x_scaled = scaler.transform(x)




predict = x_scaled @ beta

print(predict)



plt.scatter(Y, df_scaled['area'], color = 'blue')
plt.scatter(y_p, df_scaled['area'], color = 'red')

plt.ylabel('price')
plt.xlabel('area')
plt.show()


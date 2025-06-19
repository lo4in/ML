import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

path = "homework/housing-3.csv"

df = pd.read_csv(path)


df = df.replace({'yes': 0 , 'no': 1}).infer_objects(copy=False)
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
# df = df.replace({False: 0 , True: 1})





df = df.drop(columns=['furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished', 'guestroom', 'hotwaterheating', 'mainroad'])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_scaled.describe().T)
Y = df_scaled['price'].values
X = df_scaled.drop(columns = ['price']).values

N = X.shape[0]
X = np.c_[np.ones(N), X]

beta = np.linalg.inv(X.T@X)@X.T@Y

# print(beta)


# print(df_scaled['price'])

y_p = X@ beta
# print(y_p)


MAE = np.mean(np.abs(y_p - Y))



MSE = np.mean((y_p - Y)**2)




from sklearn.metrics import r2_score

r2 = r2_score(Y, y_p)

print("MAE: ", MAE)
print("MSE: ", MSE)
print("R^2:", r2)

# print(df_scaled.head())
plt.scatter(Y, df_scaled['area'], color = 'blue')
plt.scatter(y_p, df_scaled['area'], color = 'red')

plt.ylabel('price')
plt.xlabel('area')
plt.show()






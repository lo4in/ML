import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

path = 'advertising.csv'

df = pd.read_csv(path)

# print(df.head())

X = df.drop(columns = ['Sales']).values
Y = df['Sales'].values

# train_size = int(0.7* X.shape[0])
# print(train_size)

# X_train = X[:train_size]
# X_test = X[train_size:]


# ix = np.random.permutation(X.shape[0])
# X[ix]
# Y[ix]

# X_train = X[ix][:train_size]
# X_test = Y[ix][train_size:]

# Y_train, Y_test = Y[ix][:train_size], Y[ix][train_size:]
# print(X_train)


from sklearn.model_selection  import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 69)

# print(X_train[:10])

std_scal = StandardScaler()
make_poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = True)
lin_reg = LinearRegression()


X_train = std_scal.fit_transform(X_train)
X_train = make_poly.fit_transform(X_train)
lin_reg.fit(X_train, Y_train)
# print(lin_reg.score(X_train, Y_train))



y_p = lin_reg.predict(X_train)


# print(y_p)

X_test= std_scal.transform(X_test)
X_test = make_poly.transform(X_test)



print("Score for train:", lin_reg.score(X_train, Y_train))
print("Score for test:", lin_reg.score(X_test, Y_test))



from sklearn.metrics import root_mean_squared_error


print("R^2:", root_mean_squared_error(Y_train, y_p))





# print(type(X_train))
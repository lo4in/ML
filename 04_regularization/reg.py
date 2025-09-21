import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


path = "data/california_housing.csv"

df = pd.read_csv(path)


Y = df['MedHouseVal'].values
X = df.drop('MedHouseVal', axis=1).values

# print(Y, X)

from sklearn.model_selection  import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 69)



# print(X_train.shape)

# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)


from sklearn.preprocessing import StandardScaler

scal = StandardScaler()

X_train = scal.fit_transform(X_train)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=1, include_bias=False)
X_train = poly.fit_transform(X_train)

#from sklearn.linear_model import LinearRegression

#lin_reg = LinearRegression()

#lin_reg.fit(X_train, Y_train)


X_test = scal.transform(X_test)


X_test = poly.fit_transform(X_test)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.00000000000000001)
ridge.fit(X_train, Y_train)
y_p = ridge.predict(X_test)


    # print(ridge.score(X_train, Y_train))
    # print(ridge.score(X_test, Y_test))
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, y_p)

# print(r2)


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.00001)  # можно варьировать alpha
lasso.fit(X_train, Y_train)

y_pred = lasso.predict(X_test)

print("R²:", r2_score(Y_test, y_pred))
print("Коэффициенты модели:", lasso.coef_)
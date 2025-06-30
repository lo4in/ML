import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

path = 'advertising.csv'

df = pd.read_csv(path)

print(df.head())


X = df.drop(columns = ['Sales']).values
Y = df['Sales'].values

std_scal = StandardScaler()
make_poly = PolynomialFeatures(degree = 3, include_bias=False)
lin_reg = LinearRegression()

X = std_scal.fit_transform(X)
X = make_poly.fit_transform(X)


lin_reg.fit(X, Y)
y_p = lin_reg.predict(X)
print(y_p)
print(lin_reg.score)



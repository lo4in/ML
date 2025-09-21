import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "housing-2.csv"

df = pd.read_csv(path)




print(df.head(10))


x = df['area'].values
y = df['price'].values


beta_1 = (np.mean(y)*np.mean(x) - np.mean(x*y))/(np.mean(x)**2 - np.mean(x**2))
beta_0 = np.mean(y) - beta_1* np.mean(x)


y_p = beta_1 * x + beta_0



plt.scatter(x, y)
plt.scatter(x, y_p)
plt.plot(x, y_p)
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
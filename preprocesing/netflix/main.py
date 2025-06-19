import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


url = "netflix/student_scores.csv"

df = pd.read_csv(url)


def model(Hours):
    return 9.5*Hours + 5

def Plot():
    plt.scatter(df['Hours'], df['Scores'])

    plt.plot(df['Hours'], y_p)
    plt.scatter(df['Hours'], y_p)

    plt.xlabel('Hours')
    plt.ylabel('Scores')

    plt.show()




# print(y_p)
# print(y)

y_p = model(df['Hours']).values

y = df['Scores'].values

x = df['Hours'].values



epsilon_q = (y - y_p)**2
MSE = np.sum(epsilon_q)

K = np.sqrt(MSE)/len(y)


print(epsilon_q)

print(MSE)
print(K)


# beta_1 = (np.mean(y)*np.mean(x) - np.mean(x*y))/(np.mean(x)**2 - np.mean(x**2))
# beta_0 = np.mean(y) - beta_1 * np.mean(x)

# print(beta_1)
# print(beta_0)





# plt.scatter(x, y)
# plt.plot(x, y_p)
# plt.scatter(x, y_p)


# plt.xlabel('Hours')
# plt.ylabel('Scores')

#plt.show()


import statistics as stat


beta_1  = stat.covariance(x, y)/ stat.variance(x)
beta_0 = np.mean(y) - beta_1 * np.mean(x)


y_p = beta_1 * x + beta_0


plt.scatter(x, y)
plt.plot(x, y_p)
plt.scatter(x, y_p)


plt.xlabel('Hours')
plt.ylabel('Scores')

plt.show()

print(beta_1)

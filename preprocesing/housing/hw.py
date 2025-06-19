import os
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/Users/lochinbek/Desktop/BIandAI/ML/preprocesing/housing.csv")




df_encode = pd.get_dummies(df, columns=["ocean_proximity"], prefix="ocean", drop_first=False)

imputer = SimpleImputer(strategy='mean')
imputer.fit(df_encode)
a = imputer.transform(df_encode)
data1 = pd.DataFrame(data = a, columns = df_encode.columns)


x = imputer.statistics_
y = imputer.feature_names_in_


#data1['population'].hist(bins = 100)

#plt.show()
g = data1['population'].quantile(0.99)
print(g)
#d= (data1['population'] > g).sum()
data1.loc[data1['population'] > g] = data1['population'].mean()

d= (data1['population'] > g).sum()
#print(d)
#print(data1['population'].mean())


#data1['population'].hist(bins=100)
#plt.show()


#print(data1.describe())


#print(data1.columns)

select_columns = ['housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income','median_house_value']


data2 = data1[select_columns]




data_scaled = (data2 - data2.mean())/data2.std()


print(data_scaled.describe())


arr = data2.values

print(arr.std())



from sklearn.preprocessing import StandardScaler


std_scaler = StandardScaler()

std_scaler.fit(data2)



data_scaled1 = std_scaler.transform(data2)

print(data_scaled1)




data_norm = (data2 - data2.min())/(data2.max()-data2.min())

print(data_norm.describe())
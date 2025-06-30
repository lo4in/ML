import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

df  = pd.read_csv('advertising.csv')


X = df.drop(columns ='Sales').values
Y = df['Sales'].values

X = scaler.fit_transform(X)

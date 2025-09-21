import numpy as np 
from sklearn.datasets import fetch_openml
mnist =fetch_openml('mnist_784')

X = mnist.data.values()

from
from sklearn.cluster import KMeans

model = KMeans(n_clusters=10)


model.fit(X)
pred = model.predict()
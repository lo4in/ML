import numpy as np 
from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784', as_frame = False)

X = mnist.data 
y = mnist.target.astype(int)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=43)



from sklearn.svm import LinearSVC

model = LinearSVC(max_iter=2000)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(model.coef_)
print(model.intercept_)

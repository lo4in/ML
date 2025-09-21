from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
iris = load_iris(as_frame=False)
print(iris)
# print(dir(iris))
# print(iris.DESCR)

x = iris.data 
y = iris.target

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size = 0.1, random_state = 42)

std_enc = StandardScaler()
x_train = std_enc.fit_transform(x_train)
x_test = std_enc.fit_transform(x_test)

model = SVC()

model.fit(x_train, y_train)

y_pred1 = model.predict(x_train)
y_pred2 = model.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, y_pred1))
print(accuracy_score(y_test, y_pred2))

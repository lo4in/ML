from sklearn.datasets import fetch_openml 
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame = False)

index = 226
img = mnist.data[index].reshape(28, 28)
target = mnist.target[index]
print(target)
plt.imshow(img, cmap='binary')
plt.show()

X, Y = mnist.data, mnist.target
X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]

Y_train_5 = (Y_train == '5')
Y_test_5 = (Y_test == '5')


from sklearn.linear_model import SGDClassifier

sgd_clf =  SGDClassifier()
sgd_clf.fit(X_train, Y_train_5)
print(sgd_clf.score(X_test, Y_test_5))

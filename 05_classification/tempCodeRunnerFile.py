index = 226
img = mnist.data[index].reshape(28, 28)
target = mnist.target[index]
print(target)
plt.imshow(img, cmap='binary')
plt.show()

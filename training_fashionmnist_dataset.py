import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images,test_labels) = data.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress',
               'Coat','Sandal','Shirt','Sneaker','Bag','Anklel boot']
train_images = train_images/255.0
test_images = test_images/255.0

"""print(train_images[5])
plt.imshow(train_images[5],cmap=plt.cm.binary)
plt.show()"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images,train_labels)

test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Accuracy on test images = ",test_acc)
#We are going to loop through first 5 test images and show the predicted and actual result

predicton = model.predict(test_images)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: "+ class_names[test_labels[i]])
    plt.title("Prediction: "+ class_names[np.argmax(predicton[i])])
    plt.show()
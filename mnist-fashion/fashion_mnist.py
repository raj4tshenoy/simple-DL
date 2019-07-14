import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_data = train_data/255.0
test_data = test_data/255.0

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (28,28)),
        tf.keras.layers.Dense(units = 512, activation = 'relu'),
        tf.keras.layers.Dense(units = 10, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy: ',test_acc,'Test Loss: ',test_loss)

num = int(input('Enter a number b/w 0 to 9999'))
if num < 0 or num > 9999:
    print('Code cannot be executed')
    exit(0)
prediction = model.predict(test_data)
plt.imshow(test_data[num])
print(class_names[np.argmax(prediction[num])])

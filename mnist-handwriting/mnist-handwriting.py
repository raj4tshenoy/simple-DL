
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

class EndTraining(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('loss') < 0.1):
            print('\n\nRequired loss reached. Stopping training.')
            self.model.stop_training = True

#declare callback to stop training if loss is < 10%
endtraining = EndTraining()

#load data and split to train and test set
dataset = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = dataset.load_data()
print(np.shape(x_train))

x_train = x_train/255.0
x_test = x_test/255.0

#define model
model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (28,28)),
        keras.layers.Dense(units = 512, activation = 'relu', input_shape = (60000,784)),
        keras.layers.Dense(units = 10, activation = 'softmax', input_shape = (512,784))
])

#compile model
model.compile(optimizer = 'sgd',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#train model
model.fit(x_train, y_train, epochs = 7, callbacks = [endtraining])

#test model
tl, ta = model.evaluate(x_test,y_test)
print('Test Accuracy: ',ta,'Test Loss: ', tl)

#Some extras
k = 'G'
while k != 'F':
    print('\nTest the MNIST DATA:\n')
    n = int(input('Enter a number b/w 1 and 10000'))
    n = n-1
    doraemon = model.predict(x_test)
    print('\nPrediction is', np.argmax(doraemon[n]))
    plt.imshow(x_test[n])
    k = input('Enter F to stop.')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,BatchNormalization,Activation
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

# set random value for reproducibility
seed = 21

# load the data we have
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize the input
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0

# one hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# i now create the model
model = Sequential()

# alt;model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Convolution2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same'))
model.add(Activation('relu'))

# drop layers to prevent overfitting
model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

# pooling layer now !
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# to improve my network to work off
model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# i now flatten the data
model.add(Flatten())
model.add(Dropout(0.2))

# i now create the first densely connected layer
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# selection of neuron with highest (p) rem;neurons=classes=10
model.add(Dense(class_num))
model.add(Activation('softmax'))

epochs = 25
optimizer = 'adam'

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

# get to training the model
np.random.seed(seed)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy : %.2f%%" % (scores[1]*100))

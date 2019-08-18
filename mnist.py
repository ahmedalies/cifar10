from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

#load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#create augmented image generator
image_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

#fit augmented generated images to data-set
image_generator.fit(x_train)

#show sample of images
figt = plt.figure(figsize=(20, 5))
for i in range(36):
	ax = figt.add_subplot(3, 12, i+1, xticks=[], yticks=[])
	ax.imshow(np.squeeze(x_train[i]))

#rescale images pixels from [0, 255], to [0, 1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#one-hot encode labels
n_classes = len(np.unique(y_train))
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

#break training data into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

#build the CNN model
model = Sequential()
model.add(Convolution2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPool2D(pool_size=2))
model.add(Convolution2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Convolution2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

# model.summary()

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#train the model
check_pointer = ModelCheckpoint(filepath='aug_model.weights.best.hdf5', verbose=1, save_best_only=True)
model.fit_generator(image_generator.flow(x_train, y_train, batch_size=32), steps_per_epoch=x_train.shape[0] // 32, epochs=2, verbose=2, callbacks=[check_pointer], validation_data=(x_valid, y_valid))

#load the weights that yielded the best validation accuracy
model.load_weights('aug_model.weights.best.hdf5')

#evalute test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('test accuracy: {}'.format(score[1]))
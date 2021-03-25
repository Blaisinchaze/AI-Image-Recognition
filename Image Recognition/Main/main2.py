# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings
import tensorflow as tf
import numpy as np
import os
import zipfile
from tensorflow import keras

batch_size = 128
img_height = 150
img_width = 150
TRAINING_DIR = "C:/Users/Charlie/Documents/Projects/AI-Image-Recognition/Image Recognition/Main/Data/train"

VALIDATION_DIR = "C:/Users/Charlie/Documents/Projects/AI-Image-Recognition/Image Recognition/Main/Data/validation"


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAINING_DIR,
    seed=123,
    image_size=(img_width,img_height),
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VALIDATION_DIR,
    seed=123,
    image_size=(img_width,img_height),
    batch_size=batch_size
)


class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical",
                                                              input_shape=(img_width,
                                                                           img_height,
                                                                           3)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)
classes_number = 3


IMG_SHAPE=(img_width,img_height) + (3,)

model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(4, 5, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 5, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(classes_number)
])
model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=100,
                              validation_data=val_ds)

model.save("firstModelAction.h5")

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'g', label='Loss')
plt.plot(epochs, val_loss, 'y', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()

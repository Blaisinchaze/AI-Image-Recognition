import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import datetime


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


batch_stats_callback = CollectBatchStats()

# configuration
# CHANGED TO MATCH MOBILENET INPUT SIZE
image_width = 224
image_height = 224
batch_size = 32


# variables
num_classes = 0
dataset = []
labels = []


# load directories previously split with splitfolders
train_dir = "C:/Users/Charlie/Documents/Projects/AI-Image-Recognition/Image Recognition/Main/Data - Copy/train"
val_dir = "C:/Users/Charlie/Documents/Projects/AI-Image-Recognition/Image Recognition/Main/Data - Copy/validation"
test_dir = "C:/Users/Charlie/Documents/Projects/AI-Image-Recognition/Image Recognition/Main/Data - Copy/test"


# training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size)


# validation dataset (used during training)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(image_height, image_height),
    batch_size=batch_size)


# testing dataset (images not seen by the network)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(image_height, image_height),
    batch_size=batch_size)


# load class names
class_names = np.array(train_ds.class_names)
num_classes = len(class_names)


# show example images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
])


# normalization layer
normalization_layer = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(image_height, image_width),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
])


# prepare training dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)


# normalize validation and test datasets
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


# show augmented and normalized images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# download feature extractor model
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)


# define model, change output to number of classes
model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(num_classes)
])


# compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])


# store history in the variable for future plotting
# there is a callback to the class at the start to see how the model is doing for each batch
epochs = 50

history = model.fit(train_ds, epochs=epochs,
                    validation_data=val_ds,
                    callbacks=[batch_stats_callback])


# print the model summary
model.summary()


# Plot batch metrics
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(batch_stats_callback.batch_losses)
plt.show()

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(batch_stats_callback.batch_acc)
plt.show()


# plot the training history
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# prediction on testing dataset
# take a batch of images
for test_images, test_labels in test_ds.take(1):
    break


# predict on a batch and calculate class name
predicted_batch = model.predict(test_images)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]


# show tested images
plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(test_images[n])
    label = class_names[test_labels[n].numpy()]
    if label == predicted_label_batch[n]:
        plt.title(predicted_label_batch[n].title(), color="green")
    else:
        plt.title(predicted_label_batch[n].title() + " (" + label + ")", color="red")
    plt.axis('off')

_ = plt.suptitle("Model predictions")
plt.show()

# save model for future testing
model_dir = "models/saved/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(model_dir)

# load model to check if it saved properly
reloaded = tf.keras.models.load_model(model_dir)

# predict again to check if it has the same results
result_batch = model.predict(test_images)
reloaded_result_batch = reloaded.predict(test_images)

print(abs(reloaded_result_batch - result_batch).max())
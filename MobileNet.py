import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers.advanced_activations import PReLU
import numpy as np
from PIL import Image
from tensorflow.keras import mixed_precision
import os
import csv
import tensorflow_hub as hub

tf.test.gpu_device_name()
np.random.seed(0)

label_dict = {}
for i, line in enumerate(open("MedMNIST/wnids.txt", "r")):
    label_dict[line.rstrip("\n")] = i

### PARSING TRAIN/VALDIATION FILES
directory = "MedMNIST"
labeled_x_train = []
labeled_y_train = []

for i, line in enumerate(open("MedMNIST/wnids.txt", "r")):
    newDirectory = directory + "/labeled/" + line.rstrip("\n")
    for filename in os.listdir(newDirectory):
        imagePath = newDirectory + "/" + filename
        image = Image.open(imagePath).convert("RGB")
        imageData = np.asarray(image)

        labeled_x_train.append(imageData)
        num = label_dict.get(line.rstrip("\n"))
        temp = []
        for i in range(6):
            if (i == num):
                temp.append(1.)
            else:
                temp.append(0.)
        labeled_y_train.append(temp)

unlabeled_x_train = []
unlabeled_y_train = []
newDirectory = directory + "/unlabeled"
for filename in os.listdir("MedMNIST/unlabeled"):
    imagePath = newDirectory + '/' + filename
    image = Image.open(imagePath).convert("RGB")
    imageData = np.asarray(image)

    unlabeled_x_train.append(imageData)
    num = np.random.randint(6)
    temp = []
    for i in range(6):
        if (i == num):
            temp.append(1.)
        else:
            temp.append(0.)
    unlabeled_y_train.append(temp)

### PARSING TEST FILES
x_test = []
img_id = []
testPath = directory + "/test/"
for fileName in os.listdir(testPath):
    imagePath = testPath + fileName
    image = Image.open(imagePath).convert("RGB")
    imageData = np.asarray(image)
    x_test.append(imageData)
    img_id.append(fileName.replace('.jpeg', ''))

x_test = np.array(x_test)

labeled_x_train = np.array(labeled_x_train)
labeled_y_train = np.array(labeled_y_train)

unlabeled_x_train = np.array(unlabeled_x_train)
unlabeled_y_train = np.array(unlabeled_y_train)

###### Normalizing. ######
labeled_x_train = labeled_x_train / 255.0
unlabeled_x_train = unlabeled_x_train / 255.0
x_test = x_test / 255.0

labeled_x_train, labeled_x_val, labeled_y_train, labeled_y_val = train_test_split(
    labeled_x_train, labeled_y_train, test_size=0.1)

print(labeled_y_train.shape)
print(unlabeled_y_train.shape)

combined_x = np.concatenate((labeled_x_train, unlabeled_x_train), axis=0)
combined_y = np.concatenate((labeled_y_train, unlabeled_y_train), axis=0)

BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE = 500

labeled_train_dataset = tf.data.Dataset.from_tensor_slices(
    (labeled_x_train, labeled_y_train))
labeled_train_dataset = labeled_train_dataset.shuffle(
    SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
unlabeled_train_dataset = tf.data.Dataset.from_tensor_slices(
    (unlabeled_x_train, unlabeled_y_train))
unlabeled_train_dataset = unlabeled_train_dataset.shuffle(
    SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = tf.data.Dataset.zip(
    (unlabeled_train_dataset, labeled_train_dataset))

print("Finished Parsing")


class SemisupervisedModel(keras.Model):

    def __init__(self, model, entropy=False, lamda=0.1):
        super().__init__()

        self.model = model
        self.cce = tf.keras.losses.CategoricalCrossentropy()

        self.entropy = entropy
        self.lamda = lamda

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)

        self.optimizer = optimizer
        self.accuracy_tracker = keras.metrics.Accuracy()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]
        # return [self.loss_tracker]

    def call(self, images, training=True):
        return self.model(images, training=training)

    def semisupervised_loss(self, labeled_y_true, labeled_y_pred,
                            unlabeled_y_true, unlabeled_y_pred):
        loss = self.cce(labeled_y_true, labeled_y_pred)
        if self.entropy:
            loss += -self.lamda * tf.reduce_mean(
                tf.reduce_sum(
                    unlabeled_y_pred * tf.math.log(unlabeled_y_pred + 1e-12),
                    axis=1))
        return loss

    def train_step(self, data):
        (unlabeled_images, unlabeled_y_true), (labeled_images,
                                               labeled_y_true) = data

        with tf.GradientTape() as tape:
            labeled_y_pred = self.call(labeled_images, training=True)
            unlabeled_y_pred = self.call(unlabeled_images, training=True)
            loss = self.semisupervised_loss(labeled_y_true, labeled_y_pred,
                                            unlabeled_y_true, unlabeled_y_pred)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        #self.accuracy_tracker.update_state(tf.math.argmax(unlabeled_y_true, axis=1), tf.math.argmax(unlabeled_y_pred, axis=1)) #Only for if your unlabeled data has accurate labels.
        self.accuracy_tracker.update_state(
            tf.math.argmax(labeled_y_true, axis=1),
            tf.math.argmax(labeled_y_pred, axis=1))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        y_pred = self.call(images, training=False)

        loss = self.cce(labels, y_pred)
        self.loss_tracker.update_state(loss)

        self.accuracy_tracker.update_state(tf.math.argmax(labels, axis=1),
                                           tf.math.argmax(y_pred, axis=1))

        return {m.name: m.result() for m in self.metrics}


mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
feature_extractor_model = mobilenet_v2

feature_extractor_layer = hub.KerasLayer(feature_extractor_model,
                                         input_shape=(224, 224, 3),
                                         trainable=False)

semi_model = SemisupervisedModel(keras.Sequential([
    keras.layers.Resizing(224, 224),
    feature_extractor_layer,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(6, activation="softmax"),
]),
                                 entropy=True)

NUM_EPOCHS = 20

semi_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.005),
                   metrics=["accuracy"])
semi_model.build(input_shape=(None, 64, 64, 3))
semi_model.summary()

history = semi_model.fit(train_dataset,
                         epochs=NUM_EPOCHS,
                         validation_data=(labeled_x_val, labeled_y_val))

predictions = semi_model.predict(x_test, batch_size=32)
y_test = np.argmax(predictions, axis=1)
semi_model.evaluate(labeled_x_val, labeled_y_val)
combined = [[i, j] for i, j in zip(img_id, y_test)]


## Saving the results and model
with open("mobilepredictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([["image_id", "label"]])
    writer.writerows(combined)

semi_model.save("MobileNet")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 0.6])
plt.legend(loc='lower right')
plt.show()
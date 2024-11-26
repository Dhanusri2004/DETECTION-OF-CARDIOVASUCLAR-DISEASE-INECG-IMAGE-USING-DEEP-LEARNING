# importing libraries

import numpy as np

import matplotlib.pyplot as plt
import os

import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image



train_folder = "Data/"


# variables for image size
img_width = 200
img_height = 200


# Train Dataset
train_class_names = os.listdir(train_folder)
print("Train class names: %s" % (train_class_names))

x = []
y = []

for folder in os.listdir(train_folder):
    image_list = os.listdir(train_folder + "/" + folder)
    for img_name in image_list:
        # Loading images
        img = image.load_img(train_folder + "/" + folder + "/" + img_name, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        x.append(img)
        y.append(train_class_names.index(folder))

print("Preparing Training Dataset Completed.")

print("Training Dataset")

x = np.array(x)
print(x.shape)

y = to_categorical(y)  # onehot encoding of the labels
# print(y)
print(y.shape)



#https://www.kaggle.com/code/sohaibanwaar1203/dense-net-image-classification#Implementation-of-denseNet
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D


def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)
    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate
    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(filters, growth_rate, classes, dense_block_size, layers_in_block):
    input_img = Input(shape=(200, 200, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(classes, activation='softmax')(dense_x)

    return Model(input_img, output)

dense_block_size = 3
layers_in_block = 4

growth_rate = 12
classes = 4
model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
model.summary()


from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=5)


# training
batch_size = 32
epochs = 10
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
history2=model.fit(xtrain,ytrain, epochs=epochs, batch_size=batch_size, shuffle=True,validation_data=(xtest, ytest))


model.save("model.h5")


acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
epochs = range(len(acc))


plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Plot Model Loss
loss_train = history2.history['loss']
loss_val = history2.history['val_loss']
plt.plot(epochs, loss_train, label='Training Loss')
plt.plot(epochs, loss_val, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


y_pred = model.predict(xtest)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
y_test=np.argmax(ytest, axis=1)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of DenseNet')
plt.show()
import tensorflow as tf
from keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "1"


def load_cifar10_batch(cifar10_path, batch_id):
    with open(cifar10_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def train(times):
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    output = model.fit(image_train, label_train, epochs=times, batch_size=64,validation_data=(image_val,label_val), verbose=1)
    result = model.evaluate(image_test,label_test)
    show_loss_acc(output)
    print("Loss:",result[0])
    print('Accuracy:'+ str(result[1]*100)+'%')
    model.save('./weights/' + str(times) + '_times_model.h5')
    print("have been saved in weights/" + str(times)+'_times_model.h5')

def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('./result/results_tables.png', dpi=100)
    plt.show()

cifar10_path = './dataset/cifar-10-batches-py'
image_train, label_train = load_cifar10_batch(cifar10_path, 1)

for i in range(2, 6):
    features, label = load_cifar10_batch(cifar10_path, i)
    image_train, label_train = np.concatenate([image_train, features]), np.concatenate([label_train, label])

with open(cifar10_path + '/test_batch', mode='rb') as file:
    batch = pickle.load(file, encoding='latin1')
    image_test = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    label_test = batch['labels']

image_train,image_val,label_train,label_val = train_test_split(image_train,label_train,test_size=0.2)

classes = ['plane','car','bird','cat','deer',
          'dog','frog','horse','ship','truck']

dataGen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True
)

image_train = image_train/255
dataGen.fit(image_train)
image_val = image_val/255
dataGen.fit(image_val)
image_test = image_test/255

label_train = tf.keras.utils.to_categorical(label_train,10)
label_val = tf.keras.utils.to_categorical(label_val,10)
label_test = tf.keras.utils.to_categorical(label_test,10)

n = int(input("please enter train times:"))
train(n)
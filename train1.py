import tensorflow as tf
from keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

(image_train,label_train),(image_test,label_test) = tf.keras.datasets.cifar10.load_data()
image_train,image_val,label_train,label_val = train_test_split(image_train,label_train,test_size=0.2)

'''def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath,ignore_errors=True)'''

def train(epochs):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    output = model.fit(image_train, label_train, epochs=epochs, batch_size=64, validation_data=(image_val, label_val),
                       verbose=1)
    result = model.evaluate(image_test, label_test)
    print("Loss:", result[0])
    print('Accuracy:' + str(result[1] * 100) + '%')
    show_loss_acc(output)
    path1 = './weights/' + str(epochs) + '_times_model.h5'
    model.save(path1)
    print("have been saved in weight/" + str(epochs) + '_times_model.h5')

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
    path2 = './result/results_tables.png'
    plt.savefig(path2, dpi=100)
    plt.show()

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

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Package                 Version
----------------------- ------------
numpy                   1.23.5
pandas                  1.1.4
tensorflow              2.6.0

Image need resize to (224,224)
"""


# In[1]:


import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers,Input


# In[2]:


def BasicConv(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    return x


# In[3]:


def DenseBlock(x, num_convs):
    temp = []
    temp.append(x)
    for i in range(num_convs):
        y = BasicConv(x)
        temp.append(y)
        x = layers.concatenate(temp)
    return x


# In[4]:


def TransitionBlock(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 1)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    return x


# In[5]:


def DenseNet_264(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = DenseBlock(x, 6)
    x = TransitionBlock(x)
    x = DenseBlock(x, 12)
    x = TransitionBlock(x)
    x = DenseBlock(x, 64)
    x = TransitionBlock(x)
    x = DenseBlock(x, 48)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x


# In[6]:


if __name__ == '__main__':
    import glob
    
    ##data
    imgs_path = glob.glob('Osteosarcoma-UT/Data/*/*.jpg')
    all_labels_name = [img_p.split('\\')[1] for img_p in imgs_path]
    label_names = np.unique(all_labels_name)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    index_to_label = dict((index, name) for name, index in label_to_index.items())
    all_labels = [label_to_index.get(name) for name in all_labels_name]
    np.random.seed(2023)
    random_index = np.random.permutation(len(imgs_path))
    imgs_path = np.array(imgs_path)[random_index]
    all_labels = np.array(all_labels)[random_index]
    i = int(len(imgs_path)*0.8)
    train_path = imgs_path[:i]
    train_labels = all_labels[:i]
    test_path = imgs_path[i:]
    test_labels = all_labels[i:]
    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))
    
    def load_img(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image,[224,224])
        image = tf.cast(image, tf.float32)
        image = image/255.
        return image, label
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(load_img, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(load_img, num_parallel_calls=AUTOTUNE)
    BATCH_SIZE = 16
    train_ds = train_ds.repeat().shuffle(300).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)
    
    #model
    input_shape = Input([224, 224, 3])
    outputs = DenseNet_264(input_shape)
    model = models.Model(inputs=input_shape, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])
    
    #lr & early stop
    def my_lr(epoch,init_lr=0.001,attenuation_rate=0.94,attenuation_step=2):
        lr = init_lr
        lr = lr * attenuation_rate ** (epoch // attenuation_step)
        lr = max(2e-06, lr)
        return lr
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(my_lr, verbose=False)
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=15, mode='max', restore_best_weights=True)
    
    train_count = len(train_path)
    test_count = len(test_path)
    steps_per_epoch = train_count//BATCH_SIZE
    val_steps = test_count//BATCH_SIZE
    
    history = model.fit(train_ds, epochs=100,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_ds,
                    validation_steps=val_steps,
                    callbacks=[lr_callback,es_callback])


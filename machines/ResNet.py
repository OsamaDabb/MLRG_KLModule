#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorflow.keras import models,layers,losses,metrics,optimizers


# In[2]:


class Res_Unit(layers.Layer):
    def __init__(self, out_channel, use_conv1x1=False, strides=1):
        super(Res_Unit, self).__init__()
        self.Conv_1 = layers.Conv2D(out_channel/4, kernel_size=1, strides=strides, use_bias=False)
        self.BatchNorm_1 = layers.BatchNormalization(epsilon=1e-5)
        self.Conv_2 = layers.Conv2D(out_channel/4, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.BatchNorm_2 = layers.BatchNormalization(epsilon=1e-5)
        self.Conv_3 = layers.Conv2D(out_channel, kernel_size=1, strides=1, use_bias=False)
        self.BatchNorm_3 = layers.BatchNormalization(epsilon=1e-5)
        if use_conv1x1:
            self.Conv_4 = layers.Conv2D(out_channel, kernel_size=1, strides=strides, use_bias=False)
        else:
            self.Conv_4 = None
            
    def call(self, x):
        f_x = self.Conv_1(x)
        f_x = self.BatchNorm_1(f_x)
        f_x = layers.ReLU()(f_x)
        f_x = self.Conv_2(f_x)
        f_x = self.BatchNorm_2(f_x)
        f_x = layers.ReLU()(f_x)
        f_x = self.Conv_3(f_x)
        f_x = self.BatchNorm_3(f_x)
        if self.Conv_4:
            x = self.Conv_4(x)
        x = layers.Add()([x, f_x])
        x = layers.ReLU()(x)
        return x


# In[3]:


class Block_1(layers.Layer):
    def __init__(self):
        super(Block_1, self).__init__()
        self.Unit_1 = Res_Unit(256, True, 1)
        self.Unit_2 = Res_Unit(256)
        self.Unit_3 = Res_Unit(256)
        
    def call(self, x):
        x = self.Unit_1(x)
        x = self.Unit_2(x)
        x = self.Unit_3(x)
        return x


# In[4]:


class Block_2(layers.Layer):
    def __init__(self):
        super(Block_2, self).__init__()
        self.Unit_1 = Res_Unit(512, True, 2)
        self.Unit_2 = Res_Unit(512)
        self.Unit_3 = Res_Unit(512)
        self.Unit_4 = Res_Unit(512)
        self.Unit_5 = Res_Unit(512)
        self.Unit_6 = Res_Unit(512)
        self.Unit_7 = Res_Unit(512)
        self.Unit_8 = Res_Unit(512)
        
    def call(self, x):
        x = self.Unit_1(x)
        x = self.Unit_2(x)
        x = self.Unit_3(x)
        x = self.Unit_4(x)
        x = self.Unit_5(x)
        x = self.Unit_6(x)
        x = self.Unit_7(x)
        x = self.Unit_8(x)
        return x


# In[5]:


class Block_3(layers.Layer):
    def __init__(self):
        super(Block_3, self).__init__()
        self.Unit_1 = Res_Unit(1024, True, 2)
        self.Unit_2 = Res_Unit(1024)
        self.Unit_3 = Res_Unit(1024)
        self.Unit_4 = Res_Unit(1024)
        self.Unit_5 = Res_Unit(1024)
        self.Unit_6 = Res_Unit(1024)
        self.Unit_7 = Res_Unit(1024)
        self.Unit_8 = Res_Unit(1024)
        self.Unit_9 = Res_Unit(1024)
        self.Unit_10 = Res_Unit(1024)
        self.Unit_11 = Res_Unit(1024)
        self.Unit_12 = Res_Unit(1024)
        self.Unit_13 = Res_Unit(1024)
        self.Unit_14 = Res_Unit(1024)
        self.Unit_15 = Res_Unit(1024)
        self.Unit_16 = Res_Unit(1024)
        self.Unit_17 = Res_Unit(1024)
        self.Unit_18 = Res_Unit(1024)
        self.Unit_19 = Res_Unit(1024)
        self.Unit_20 = Res_Unit(1024)
        self.Unit_21 = Res_Unit(1024)
        self.Unit_22 = Res_Unit(1024)
        self.Unit_23 = Res_Unit(1024)
        self.Unit_24 = Res_Unit(1024)
        self.Unit_25 = Res_Unit(1024)
        self.Unit_26 = Res_Unit(1024)
        self.Unit_27 = Res_Unit(1024)
        self.Unit_28 = Res_Unit(1024)
        self.Unit_29 = Res_Unit(1024)
        self.Unit_30 = Res_Unit(1024)
        self.Unit_31 = Res_Unit(1024)
        self.Unit_32 = Res_Unit(1024)
        self.Unit_33 = Res_Unit(1024)
        self.Unit_34 = Res_Unit(1024)
        self.Unit_35 = Res_Unit(1024)
        self.Unit_36 = Res_Unit(1024)
        
    def call(self, x):
        x = self.Unit_1(x)
        x = self.Unit_2(x)
        x = self.Unit_3(x)
        x = self.Unit_4(x)
        x = self.Unit_5(x)
        x = self.Unit_6(x)
        x = self.Unit_7(x)
        x = self.Unit_8(x)
        x = self.Unit_9(x)
        x = self.Unit_10(x)
        x = self.Unit_11(x)
        x = self.Unit_12(x)
        x = self.Unit_13(x)
        x = self.Unit_14(x)
        x = self.Unit_15(x)
        x = self.Unit_16(x)
        x = self.Unit_17(x)
        x = self.Unit_18(x)
        x = self.Unit_19(x)
        x = self.Unit_20(x)
        x = self.Unit_21(x)
        x = self.Unit_22(x)
        x = self.Unit_23(x)
        x = self.Unit_24(x)
        x = self.Unit_25(x)
        x = self.Unit_26(x)
        x = self.Unit_27(x)
        x = self.Unit_28(x)
        x = self.Unit_29(x)
        x = self.Unit_30(x)
        x = self.Unit_31(x)
        x = self.Unit_32(x)
        x = self.Unit_33(x)
        x = self.Unit_34(x)
        x = self.Unit_35(x)
        x = self.Unit_36(x)
        return x


# In[6]:


class Block_4(layers.Layer):
    def __init__(self):
        super(Block_4, self).__init__()
        self.Unit_1 = Res_Unit(2048, True, 2)
        self.Unit_2 = Res_Unit(2048)
        self.Unit_3 = Res_Unit(2048)
        
    def call(self, x):
        x = self.Unit_1(x)
        x = self.Unit_2(x)
        x = self.Unit_3(x)
        return x


# In[7]:


class ResNet_152(models.Model):
    def __init__(self):
        super(ResNet_152, self).__init__()
        self.Conv = layers.Conv2D(64, kernel_size=7, strides=2, padding='SAME', use_bias=False)
        self.BatchNorm = layers.BatchNormalization(epsilon=1e-5)
        self.MaxPool = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')
        
        self.Block_1 = Block_1()
        self.Block_2 = Block_2()
        self.Block_3 = Block_3()
        self.Block_4 = Block_4()
        
        self.AvgPool = layers.GlobalAveragePooling2D()
        self.Dense = layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = layers.ReLU()(x)
        x = self.MaxPool(x)
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x = self.Block_4(x)
        x = self.AvgPool(x)
        x = self.Dense(x)
        return x


# In[8]:


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
    model = ResNet_152()
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


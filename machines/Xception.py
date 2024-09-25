#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""
Package                 Version
----------------------- ------------
numpy                   1.23.5
pandas                  1.1.4
tensorflow              2.6.0

Image need resize to (299,299)
"""


# In[1]:


import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers


# In[2]:


class Xception(models.Model):
    def __init__(self):
        super(Xception, self).__init__()

    def build(self, input_shape):
        self.Conv_1 = BasicConv(32, (3,3), (2,2))
        self.Conv_2 = BasicConv(64, (3,3))
        
        self.ResConv = layers.Conv2D(filters=128,kernel_size=(1,1),strides=(2,2),padding='same',use_bias=False)
        self.ResBatchNormal = layers.BatchNormalization()
        self.SepConv_1 = layers.SeparableConv2D(filters=128, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_1 = layers.BatchNormalization()
        self.Relu = layers.ReLU()
        self.SepConv_2 = layers.SeparableConv2D(filters=128, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_2 = layers.BatchNormalization()
        self.MaxPool = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')
        
        self.ResUnit_2 = ResidualUnit(256)
        self.ResUnit_3 = ResidualUnit(728)
        
        self.MidUnit_1 = MiddleFlowUnit()
        self.MidUnit_2 = MiddleFlowUnit()
        self.MidUnit_3 = MiddleFlowUnit()
        self.MidUnit_4 = MiddleFlowUnit()
        self.MidUnit_5 = MiddleFlowUnit()
        self.MidUnit_6 = MiddleFlowUnit()
        self.MidUnit_7 = MiddleFlowUnit()
        self.MidUnit_8 = MiddleFlowUnit()
        
        self.ExitFlow = ExitFlow()
        self.AvePool = layers.GlobalAveragePooling2D()
        self.Dropout = layers.Dropout(0.5)
        self.fc_1 = layers.Dense(1, activation='sigmoid')
        super(Xception,self).build(input_shape)

    def call(self, x):
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        
        residual = self.ResConv(x)
        residual = self.ResBatchNormal(residual)
        x = self.SepConv_1(x)
        x = self.SepBatchNormal_1(x)
        x = self.Relu(x)
        x = self.SepConv_2(x)
        x = self.SepBatchNormal_2(x)
        x = self.MaxPool(x)
        x = layers.add([x, residual])
        
        x = self.ResUnit_2(x)
        x = self.ResUnit_3(x)
        
        x = self.MidUnit_1(x)
        x = self.MidUnit_2(x)
        x = self.MidUnit_3(x)
        x = self.MidUnit_4(x)
        x = self.MidUnit_5(x)
        x = self.MidUnit_6(x)
        x = self.MidUnit_7(x)
        x = self.MidUnit_8(x)
        
        x = self.ExitFlow(x)
        x = self.AvePool(x)
        x = self.Dropout(x)
        x = self.fc_1(x)
        
        return x


# In[3]:


class BasicConv(layers.Layer):
    def __init__(self,filters,kernel_size,strides=(1,1),use_bias=False):
        super(BasicConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        
    def build(self,input_shape):
        self.Conv = layers.Conv2D(filters = self.filters, kernel_size = self.kernel_size, strides = self.strides, use_bias = self.use_bias)
        self.BatchNormal = layers.BatchNormalization()
        self.Relu = layers.ReLU()
        
    def call(self, x):
        x = self.Conv(x)
        x = self.BatchNormal(x)
        x = self.Relu(x)
        return x


# In[4]:


class ResidualUnit(layers.Layer):
    def __init__(self, filters):
        super(ResidualUnit, self).__init__()
        self.filters = filters
        
    def build(self, input_shape):
        self.ResConv = layers.Conv2D(filters=self.filters,kernel_size=(1,1),strides=(2,2),padding='same',use_bias=False)
        self.ResBatchNormal = layers.BatchNormalization()
        self.Relu_1 = layers.ReLU()
        self.SepConv_1 = layers.SeparableConv2D(filters=self.filters, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_1 = layers.BatchNormalization()
        self.Relu_2 = layers.ReLU()
        self.SepConv_2 = layers.SeparableConv2D(filters=self.filters, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_2 = layers.BatchNormalization()
        self.MaxPool = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')
        
    def call(self, x):
        residual = self.ResConv(x)
        residual = self.ResBatchNormal(residual)
        x = self.Relu_1(x)
        x = self.SepConv_1(x)
        x = self.SepBatchNormal_1(x)
        x = self.Relu_2(x)
        x = self.SepConv_2(x)
        x = self.SepBatchNormal_2(x)
        x = self.MaxPool(x)
        x = layers.add([x, residual])
        return x


# In[5]:


class MiddleFlowUnit(layers.Layer):
    def __init__(self):
        super(MiddleFlowUnit, self).__init__()
        
    def build(self, input_shape):
        self.Relu_1 = layers.ReLU()
        self.SepConv_1 = layers.SeparableConv2D(filters=728, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_1 = layers.BatchNormalization()
        self.Relu_2 = layers.ReLU()
        self.SepConv_2 = layers.SeparableConv2D(filters=728, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_2 = layers.BatchNormalization()
        self.Relu_3 = layers.ReLU()
        self.SepConv_3 = layers.SeparableConv2D(filters=728, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_3 = layers.BatchNormalization()
        
    def call(self, x):
        residual = x
        x = self.Relu_1(x)
        x = self.SepConv_1(x)
        x = self.SepBatchNormal_1(x)
        x = self.Relu_2(x)
        x = self.SepConv_2(x)
        x = self.SepBatchNormal_2(x)
        x = self.Relu_3(x)
        x = self.SepConv_3(x)
        x = self.SepBatchNormal_3(x)
        x = layers.add([x, residual])
        return x


# In[6]:


class ExitFlow(layers.Layer):
    def __init__(self):
        super(ExitFlow, self).__init__()
        
    def build(self, input_shape):
        self.ResConv = layers.Conv2D(filters=1024,kernel_size=(1,1),strides=(2,2),padding='same',use_bias=False)
        self.ResBatchNormal = layers.BatchNormalization()
        self.Relu_1 = layers.ReLU()
        self.SepConv_1 = layers.SeparableConv2D(filters=728, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_1 = layers.BatchNormalization()
        self.Relu_2 = layers.ReLU()
        self.SepConv_2 = layers.SeparableConv2D(filters=1024, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_2 = layers.BatchNormalization()
        self.MaxPool = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')
        
        self.SepConv_3 = layers.SeparableConv2D(filters=1536, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_3 = layers.BatchNormalization()
        self.Relu_3 = layers.ReLU()
        self.SepConv_4 = layers.SeparableConv2D(filters=2048, kernel_size=(3,3),padding='same',use_bias=False)
        self.SepBatchNormal_4 = layers.BatchNormalization()
        self.Relu_4 = layers.ReLU()
        
    def call(self, x):
        residual = self.ResConv(x)
        residual = self.ResBatchNormal(residual)
        x = self.Relu_1(x)
        x = self.SepConv_1(x)
        x = self.SepBatchNormal_1(x)
        x = self.Relu_2(x)
        x = self.SepConv_2(x)
        x = self.SepBatchNormal_2(x)
        x = self.MaxPool(x)
        x = layers.add([x, residual])
        
        x = self.SepConv_3(x)
        x = self.SepBatchNormal_3(x)
        x = self.Relu_3(x)
        x = self.SepConv_4(x)
        x = self.SepBatchNormal_4(x)
        x = self.Relu_4(x)
        return x


# In[7]:


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
        image = tf.image.resize(image,[299,299])
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
    model = Xception()
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


import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers,Input
from machines.Xception import Xception
from machines.ResNet import ResNet_152
from machines.DenseNet import DenseNet_264

class ImageModels:
    def __init__(self, model_type):
        if model_type == 'Xception':
            self.model = Xception()
        elif model_type == 'Resnet':
            self.model = ResNet_152()
        elif model_type == 'Densenet':
            input_shape = Input([224, 224, 3])
            outputs = DenseNet_264(input_shape)
            self.model = models.Model(inputs=input_shape, outputs=outputs)
        else:
            raise ValueError('Model type not recognized. Choose Xception, Resnet or Densenet')
    
    def get_model(self):
        return self.model
    
class BinaryModels:
    def __init__(self, model_type, X_train=None, y_train=None, kernel='linear', gamma='auto', n_neighbors=5, weights='uniform',
                 penalty = 'l2', solver = 'liblinear'):
        model_type = model_type.lower()
        if model_type == 'binary svm':
            from sklearn import svm
            clf = svm.SVC(kernel=kernel, gamma=gamma, probability=True)
            self.model = clf.fit(X_train,y_train)
        elif model_type == 'binary knn':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'auto')
            self.model = clf.fit(X_train,y_train)
        elif model_type == 'binary logistic regression':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(penalty = 'l2', solver = 'liblinear')    
            self.model = clf.fit(X_train, y_train)
        elif model_type == 'binary deepnet':
            input_shape = X_train.shape[1]
            input_shape = Input(input_shape)
            outputs = DeepNet(input_shape)
            self.model = models.Model(inputs=input_shape, outputs=outputs)
        else:
            raise ValueError('Model type not recognized. Choose svm, knn, logistic or deepnet')

    def get_model(self):
        return self.model
            
            
def DeepNet(x):
    x = layers.Dense(1046)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x

# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers

# class Machine:
#     def __init__(self, model_name='xception', learning_rate=0.001):
#         self.model_name = model_name
#         self.learning_rate = learning_rate
#         self.model = self.load_model()

#     def load_model(self):
#         """
#         Load the specified pre-trained model and compile it with a customizable learning rate.
#         """
#         if self.model_name.lower() == 'xception':
#             base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
#         elif self.model_name.lower() == 'resnet':
#             base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#         elif self.model_name.lower() == 'densenet':
#             base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#         else:
#             raise ValueError("Unsupported model. Please choose 'xception', 'resnet', or 'densenet'.")

#         # Freeze the base_model
#         base_model.trainable = False

#         # Create new model on top
#         inputs = tf.keras.Input(shape=(299, 299, 3))
#         x = base_model(inputs, training=False)
#         x = layers.GlobalAveragePooling2D()(x)
#         outputs = layers.Dense(1, activation='sigmoid')(x)
#         model = models.Model(inputs, outputs)

#         # Compile the model
#         optimizer = optimizers.Adam(learning_rate=self.learning_rate)
#         model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#         return model

#     def train_model(self, train_data, validation_data, epochs):
#         """
#         Train the model on the provided data.

#         Parameters:
#         - train_data: Training data.
#         - validation_data: Validation data.
#         - epochs: Number of epochs to train for.
#         """
#         return self.model.fit(train_data, validation_data=validation_data, epochs=epochs)

#     def set_learning_rate(self, learning_rate):
#         """
#         Update the learning rate of the model's optimizer.
#         """
#         self.learning_rate = learning_rate
#         self.model.optimizer.lr = learning_rate

# # Example usage
# machine = Machine(model_name='xception', learning_rate=0.001)
# # machine.train_model(train_data, validation_data, epochs=10)
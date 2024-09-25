 Image models are Done. When we have the current raw data accuracy can you share here YUCEN! 

In Machines.py we could get 3 different Image Models: Xception, Resnet and Densenet.

Input shape: (299,299,3) --> Xception; (224,224,3) --> Resnet and Densenet
Yucen our input will be (512,512,1) and (512,512,3) it can vary we should talk with features team.

Input of ImageModels is model_type(contains:'Xception','Resnet','Densenet'). Eg: model = ImageModels('Xception').get_model()

Next task: finding a better optimizer that is less sensitive to the learning rate(reason: wanna fewer parameters)

We have currently 15-20 parameters for meta-learning and lets plan to apply random search.

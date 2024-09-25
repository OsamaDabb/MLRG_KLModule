# This is completely remaking the Image Pipeline Functions module with the new method described in image_pipeline_testing_2.
# once finished, it will be transfered to image_pipeline_functions to be implemented with the actual main and run function

# Template for function comments:

###########################################
# X function
# imports:
# Purpose: 
# input: 
#       example: 
# output:
#       example: 
###########################################


##############################################################################################
##############################################################################################
#           Constants
##############################################################################################
##############################################################################################
# This is the param grid that's used for making random augmentation parameters
# Numpy linspace? np.linspace to make the same arrays but with a function to simplify and make easier to change
PARAM_GRID = {
        'shear_factor': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'shear_prop': [0.1, 0.2, 0.3],
        'crop_scale_factor': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        'crop_scale_prop': [0.1, 0.2, 0.3],
        'flip_code': [-1, 0, 1],
        'flip_prop': [0.1, 0.2, 0.3],
        'rotation_angle': [0, 90, 180, 270],
        'rotate_prop': [0.1, 0.2, 0.3],
        'color_number': [0, 1, 2, 3, 4, 5],
        'blur_param': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'blur_prop': [0.1, 0.2, 0.3]
    }

# increasing batch size doesn't make it that much faster, probably best to hover around 50?
BATCH_SIZE = 10

##############################################################################################
##############################################################################################
#           Imports
##############################################################################################
##############################################################################################
# When adding imports, please add which functions the packages are used in for the comments

# Imports for tensorflow and run_model function

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from tensorflow.keras.applications import Xception, ResNet152V2, DenseNet201
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import RMSprop

#Import os for folder stuff
import os

# Import shutil for making folders
import shutil

# import random for split_images_to_train_test
import random

# Import metrics and plot for ROC-AUC curve for the run_model function
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import numpy as np

# import augmentation functions for augmentation function
from data_augmentation.image_processor_all import ImageProcessor,ImageProcessor2,augment_images_with_proportions_2,augment_images_with_proportions,combined_augment_function

# Import feature functions for features
from features.image_features import *
# GPUS

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


##############################################################################################
##############################################################################################
#           Image Dataframe Functions
##############################################################################################
##############################################################################################
# These are all the functions that are used in the image dataframe pipeline . 


# Folder Functions #
####################


###########################################
# remove_images_from_folder function
# Purpose: utility function to remove images from folders
# input: folder path
#       example: "/projectnb/mlresearch/test/ani/cancer/data/Image/test/cancer"
# output: None
###########################################

def remove_images_from_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file in the folder
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        # Check if the file is a regular file and ends with an image extension
        if os.path.isfile(file_path) and file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            os.remove(file_path)  # Remove the file

    print(f"All images removed from the folder: {folder_path}")

###########################################
# make_temp_folder function
# Purpose: utility function to make a temporary folder with a specified number of images.
#         Checks if the folder exists, then makes the folder if it doesn't. if the folder exists, remove images from folder using above function.
#         Is used in image df pipeline in order to optimize augmentation, so you don't have to augment all the images in the folder
# input: folder path
#       example: "/projectnb/mlresearch/test/ani/cancer/data/Image/test/cancer"
# output: temp folder
#       example: "/projectnb/mlresearch/test/ani/cancer/data/Image/test/cancer_temp"
###########################################

def make_temp_folder(folder, num_images):
    folder_temp = folder + "_temp"
    if not os.path.exists(folder_temp):
        os.makedirs(folder_temp)
    else:
        remove_images_from_folder(folder_temp)
    
    image_files = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]
    for file_name in image_files[:num_images]:
        source_file = os.path.join(folder, file_name)
        destination_file = os.path.join(folder_temp, file_name)
        shutil.copy(source_file, destination_file)
    return folder_temp

###########################################
# count_images_in_folder function
# Purpose: utility function to count the number of images in a folder.
#          is used to select the lower number of images between two classes so the number of training data is balanced
#          looks through all the files in the given folder and makes a list of all files that end in .jpg, .jpeg, .png, .gif
# input: folder path
#       example: "/projectnb/mlresearch/test/ani/cancer/data/Image/test/cancer"
# output: number of images
#       example: like 3 i guess idk
###########################################

def count_images_in_folder(folder_path):
    # List files in the folder
    files = os.listdir(folder_path)

    # Filter only image files (assuming extensions like .jpg, .png, etc.)
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Count the number of image files
    num_images = len(image_files)
    return num_images




# Other Functions  #
####################

###########################################
# get_augmentation_parameters function
# Purpose: function to create parameters for augmentation function.
#          For the most part the random parameters are only used for testing/benchmarking, when using the GUI
#          the user params are used.
# inputs: random_params (boolean), user_params (array)
#       example: get_augmentation_parameters(True, user_params=None)
#                   This gets you random parameters, otherwise you can enter your own array like:
#                get_augmentation_parameters(False, user_params=[0.1, 0.2, 50, ...])
# output: param_dict (dictionary)
#       example: ["shear_factor": 0.1, ...]
###########################################

def get_augmentation_parameters(random_params, user_params=None):
    # randomly sample for each np.linspace 
    param_combinations = [dict(zip(PARAM_GRID.keys(), values)) for values in itertools.product(*PARAM_GRID.values())]
    # print("param_combinations starting...")
    for param_dict in param_combinations:
        param_dict['color_prop'] = 1 - (param_dict['rotate_prop'] + param_dict['flip_prop'] + param_dict['crop_scale_prop'])
    # print("param_combinations finished!")
    if random_params:
        param_dict = random.choice(param_combinations)
        # print("Selected random param_dict:")
        # print(param_dict)
    else:
        if user_params == None:
            param_dict = param_combinations[0]
            print("Selected first param_dict:")
            print(param_dict)
        else:
            param_dict = {}
            for key, value in zip(PARAM_GRID.keys(), user_params):
                param_dict[key] = value
            param_dict['color_prop'] = 1 - (param_dict['rotate_prop'] + param_dict['flip_prop'] + param_dict['crop_scale_prop'])
            print("Selected user-defined param_dict:")
            print(param_dict)
    return param_dict

###########################################
# put_images_in_df function
# Purpose: Function to be used in make_image_df.
# input: images (list of paths to images), tumor label (0 or 1 for class)
#       example: put_images_in_df(["/projectnb/mlresearch/test/ani/cancer/data/Image/oral_normal/oral_normal_0002.jpg", ...],
#                tumor_label=0)
# output: df (pandas dataframe)
#       example: Image Array: [[[255, 255, 255], [255, 255, 255], ...        Tumor: 0
###########################################

def put_images_in_df(images, tumor_label):
    image_arrays = []
    for image_path in images:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            # Expand dimensions to make it compatible with conv2d
            img_array = np.expand_dims(img_array, axis=0)
            image_arrays.append(img_array)
    data = {'Image_Array': image_arrays, 'Tumor': [tumor_label] * len(images)}
    df = pd.DataFrame(data)
    return df

###########################################
# make_image_df function
# Purpose: Take a folder of images and turn them into a pandas dataframe with 2 columns, the Image Array and the class
# input: folder (string of path), tumor label (0 for healthy, 1 for cancer), number of images (int)
#       example: make_image_df(folder1_augmented, number_images=num_images * 4, tumor_label=0)
# output: df (pandas dataframe)
#       example: same output as put_images_in_df function
###########################################

def make_image_df(folder, tumor_label, number_images):
    images = glob.glob(folder + "/*.jpg")
    df = put_images_in_df(images[:number_images], tumor_label)
    return df
##############################################################################################
##############################################################################################
#           Machine Functions
##############################################################################################
##############################################################################################

# These are all the functions that are used in the machine pipeline. 

def add_singleton_to_tensor(tensor):
    rank = tf.rank(tensor)
    if rank == 1:
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.expand_dims(tensor, axis=0)
    elif rank == 2:
        tensor = tf.expand_dims(tensor, axis=0)
    return tensor

def add_singleton(tensor_batch):
    return tf.map_fn(add_singleton_to_tensor, tensor_batch)

def tensor_shape(tensor):
    """
    Returns the shape of a tensor.
    """
    return tf.shape(tensor)


def make_machine(machine_name, input_shape, lr):
    """
    Constructs a machine (model) with the given input shape.
    """
    if machine_name.lower() == "image xception":
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))

    elif machine_name.lower() == "image densenet201":
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))

    elif machine_name.lower() == "image resnet152":
        base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))
    else:
        raise ValueError(f"Unknown machine name: {machine_name}")
    
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=lr),
                  metrics=['accuracy'])
    return model


###########################################
# run_model function
# Purpose: This is the main function that takes in a model and uses the training and testing folders made
#          in order to actually train that model.
# input: * model (string of which model you want to use)
#            example:  "image xception"
#        * train_folder, test_folder (folder paths where training and testing data is kept)
#            example:  "/projectnb/mlresearch/test/ani/cancer/data/Image/test"
#        * epochs (int, number of times you want the model to learn)
#            example:  like 3 again its just an int choose your favorite number
#        * lr (float, basically how much you want it to jump around)
#            example:  0.01, or 0.001, or 0.0001, etc.
# output: metrics (dictionary)
#       example: {'loss': 0.6956377625465393, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
###########################################


# NOTE: Does not run if there are too few images, as there will be an error where there are technically zero validation steps

def run_model(machine, train_folder, test_folder, epochs, lr, features):
    # train_count = len(train_ds)
    # test_count = len(test_ds)
    # val_steps = test_count//BATCH_SIZE
    # steps_per_epoch = train_count//BATCH_SIZE

    if features == False:

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

    else: 
    
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            # 
            preprocessing_function=preprocessing_function
            )
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=preprocessing_function
            )

    print(train_folder)
    print(test_folder)

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(512, 512),
        batch_size=BATCH_SIZE,
        class_mode='binary')  

    validation_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(512, 512),
        batch_size=BATCH_SIZE,
        class_mode='binary')


    # input_shape = (13, 1, 1)
    model = make_machine(machine, (512,512,3), lr)



    print("Length of train_generator:", len(train_generator))
    print("Length of validation_generator:", len(validation_generator))
    
    history = model.fit(train_generator, epochs=epochs,
                        steps_per_epoch=len(train_generator) //BATCH_SIZE,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator) // BATCH_SIZE)
    # model.fit(
    #     train_generator,
    #     steps_per_epoch=2000 // batch_size,
    #     epochs=10,
    #     validation_data=validation_generator,
    #     validation_steps=800 // batch_size)
    # model.save_weights('model_weights.h5')
    loss, accuracy = model.evaluate(validation_generator)
   # Collect true labels and predictions
    y_true = []
    y_pred = []

    # Iterate through the validation generator
    for i in range(len(validation_generator)):
        x, y = validation_generator[i]
        preds = model.predict(x)
        y_true.extend(y)
        y_pred.extend(preds)

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate precision, recall, F1-score, confusion matrix, and ROC-AUC
    precision = precision_score(y_true, y_pred_binary, average='weighted')
    recall = recall_score(y_true, y_pred_binary, average='weighted')
    f1 = f1_score(y_true, y_pred_binary, average='weighted')
    cm = confusion_matrix(y_true, y_pred_binary)
    report = classification_report(y_true, y_pred_binary, target_names=['Class 0', 'Class 1'])
    roc_auc = roc_auc_score(y_true, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print("ROC-AUC Score:", roc_auc)
    print({'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, "ROC-AUC Score" : roc_auc})
    return {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, "ROC-AUC Score" : roc_auc}

    # Generate ROC-AUC plot
    # np.savez('roc_data.npz', fpr=fpr, tpr=tpr, roc_auc=roc_auc)


###########################################
# split_images_to_train_test function
# Purpose: Function that allows us to split the images from a main image folder into a train folder and test folder.
#          These train and test folders are what are input into the run_model function
# input: * source_folder (folder path where you keep the data you want to split into training and test data)
#            example:  "/projectnb/mlresearch/test/ani/cancer/data/Image/oral_normal_augmented"
#        * train_folder, test_folder (folder paths where training and testing data is kept)
#            example:  "/projectnb/mlresearch/test/ani/cancer/data/Image/test"
#        * split ratio (float, what percent do you want to go into the train_folder)
#            example:  0.8
# output: None
###########################################

def split_images_to_train_test(source_folder, train_folder, test_folder, split_ratio=0.8):
    # Create train and test folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of all images in the source folder
    images = os.listdir(source_folder)

    # Shuffle the list of images
    random.shuffle(images)

    # Calculate the number of images for the train and test sets
    num_train = int(len(images) * split_ratio)
    num_test = len(images) - num_train

    # Copy images to train folder
    for img_name in images[:num_train]:
        src_path = os.path.join(source_folder, img_name)
        dst_path = os.path.join(train_folder, img_name)
        shutil.copyfile(src_path, dst_path)

    # Copy images to test folder
    for img_name in images[num_train:]:
        src_path = os.path.join(source_folder, img_name)
        dst_path = os.path.join(test_folder, img_name)
        shutil.copyfile(src_path, dst_path)


##############################################################################################
##############################################################################################
#           Pipelines
##############################################################################################
##############################################################################################
# These are the pipelines that are called in the run_function.py


###########################################
# pipeline_augmentation function
# Purpose: This function takes a folder and then augments it so that you are able to gain more images, and thus better results with less data.
#          Previously called pipeline_image_df, but image dataframes are no longer required for the new run_model function, so name changed accordingly.
#          Only called in run function when augmentation in the main function dictionary is True (boolean) 
# input: * folder_normal, folder_cancer (folder paths where you keep the image data)
#            example:  "/projectnb/mlresearch/test/ani/cancer/data/Image/oral_normal", "/projectnb/mlresearch/test/ani/cancer/data/Image/oral_scc"
#        * random_params (Boolean, do you select random parameters from the get_augmentation_parameters function)
#        * user_params (list, which user parameters are input, only used if random_params == False)
#            example:  False, user_params=[0.1, 0.2, 50, ...]
#        * num_images (int, number of images you want to select, none if you want to use all)
#            example:  num_images = 300
# output: None
###########################################


# def pipeline_image_df(folder1, folder2, augmentation, random_params, user_params, num_images = 300):
def pipeline_image_df(folder_normal, folder_cancer, augmentation, random_params, user_params, num_images = None):
    print(augmentation)
    #TODO: Make the whole make/remove temporary folders a function
    # FIXED
    if num_images == None:
        num_images_normal = count_images_in_folder(folder_normal)
        num_images_scc = count_images_in_folder(folder_cancer)
        num_images = min(num_images_normal,num_images_scc)

    #UNCOMMENT LATER, FOR BENCHMARKING TESTING
    folder_normal_temp = make_temp_folder(folder_normal, num_images)
    folder_cancer_temp = make_temp_folder(folder_cancer, num_images)

    if augmentation == True:
        # This is stupid
        # At some point I have to clean this up
        # Instead of choosing from a dictionary in the get_augmentation_params, just enter the user_params as a dictionary and use it?
        # TODO: Temp folder is unnecessary if num_images = None
        # TODO: Fix get_augmentation_parameters function
        if random_params == False:
            params = user_params
        else:
            params = get_augmentation_parameters(random_params, user_params)
        folder_normal_augmented = folder_normal + "_augmented"
        folder_cancer_augmented = folder_cancer + "_augmented"

        if not os.path.exists(folder_normal_augmented):
            os.makedirs(folder_normal_augmented)
        else:
            remove_images_from_folder(folder_normal_augmented)
        if not os.path.exists(folder_cancer_augmented):
            os.makedirs(folder_cancer_augmented)
        else:  
            remove_images_from_folder(folder_cancer_augmented)
        
        #FOR LUNG BEHCMARKING TEMP TESTING UNCOMMENT LATER
        # if not os.path.exists(folder_normal_augmented) and not os.path.exists(folder_cancer_augmented):
        print("normal_df Augmentation")
        print("augment function")
        combined_augment_function(folder_normal_temp, folder_normal_augmented, **params)
        image_count = count_images_in_folder(folder_normal_augmented)
        print(f"Number of images in normal augmented folder: {image_count}")

        print("scc_df Augmentation")
        print("augment function")
        combined_augment_function(folder_cancer_temp, folder_cancer_augmented, **params)
        image_count = count_images_in_folder(folder_cancer_augmented)
        print(f"Number of images in scc augmented folder: {image_count}")
        
        image_count = count_images_in_folder(folder_normal_augmented)
        print(f"Number of images in normal augmented folder: {image_count}")
        image_count = count_images_in_folder(folder_cancer_augmented)
        print(f"Number of images in scc augmented folder: {image_count}")
        return params
    else:
        params = {'angle': False, 'flip_code': False, 'crop_size': False, 'shear_factor': False, 'rotate_prop':False, 'flip_prop':False, 'crop_scale_prop': False, 'color_number': False, 'kernel_size': False, 'segment_prop': False}
        return params


# This needs to be changed, apparently we need to do feature extractors in the generator itself????
# Ignore features for now, get the other stuff working.
# def pipeline_features(normal_df, scc_df, participation_factors, params = None):
#     normal_images = normal_df['Image_Array']
#     print("Starting image features")
#     normal_features_list = Image_Features.generate_image_features(normal_images, participation_factors, None)
#     print("Image features done!")
#     normal_features_df = pd.DataFrame(normal_features_list)
#     normal_df = pd.concat([normal_df, normal_features_df], axis=1)
#     scc_images = scc_df['Image_Array']
#     scc_features_list = Image_Features.generate_image_features(scc_images, participation_factors, None)
#     scc_features_df = pd.DataFrame(scc_features_list)
#     scc_df = pd.concat([scc_df, scc_features_df], axis=1)
#     params.update(participation_factors)
#     # Honestly I should really just stop this whole updating parameters thing, it gets dealt with in the run function
#     return normal_df, scc_df, params


###########################################
# pipeline_machine function
# Purpose: put the images in the folders given into different folders for testing and training, and then 
#          put those folders into the run function. Augmentation check is done in run_function
# input: * folder_normal, folder_cancer (folder paths where you keep the image data)
#            example:  "/projectnb/mlresearch/test/ani/cancer/data/Image/oral_normal", "/projectnb/mlresearch/test/ani/cancer/data/Image/oral_scc"
#        * random_params (Boolean, do you select random parameters from the get_augmentation_parameters function)
#        * user_params (list, which user parameters are input, only used if random_params == False)
#            example:  False, user_params=[0.1, 0.2, 50, ...]
#        * num_images (int, number of images you want to select, none if you want to use all)
#            example:  num_images = 300
#        * model (string of which model you want to use)
#            example:  "image xception"
#        * epochs (int, number of times you want the model to learn)
#            example:  like 3 again its just an int choose your favorite number
#        * lr (float, basically how much you want it to jump around)
#            example:  0.01, or 0.001, or 0.0001, etc.
# output: metrics (dictionary)
#       example: {'loss': 0.6956377625465393, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
###########################################


def pipeline_machine(folder_normal, folder_cancer, model, epochs, lr, features, params = None):
    # Empty array for params and metrics at the end
    train_folder =  "/projectnb/mlresearch/test/ani/cancer/data/Image/train/"
    test_folder = "/projectnb/mlresearch/test/ani/cancer/data/Image/test/"

    normal_train_folder = train_folder + "normal"
    normal_test_folder = test_folder + "normal"
    remove_images_from_folder(normal_train_folder)
    remove_images_from_folder(normal_test_folder)
    split_images_to_train_test(folder_normal, normal_train_folder, normal_test_folder)

    cancer_train_folder = train_folder + "cancer"
    cancer_test_folder = test_folder + "cancer"
    remove_images_from_folder(cancer_train_folder)
    remove_images_from_folder(cancer_test_folder)
    split_images_to_train_test(folder_cancer, cancer_train_folder, cancer_test_folder)
    number_of_images = count_images_in_folder(folder_normal)
    print(f"Number of images in the folder: {number_of_images}")
    number_of_images = count_images_in_folder(folder_cancer)
    print(f"Number of images in the folder: {number_of_images}")
    number_of_images = count_images_in_folder(cancer_train_folder)
    print(f"Number of images in the folder: {number_of_images}")
    number_of_images = count_images_in_folder(cancer_test_folder)
    print(f"Number of images in the folder: {number_of_images}")
    number_of_images = count_images_in_folder(normal_train_folder)
    print(f"Number of images in the folder: {number_of_images}")
    number_of_images = count_images_in_folder(normal_test_folder)
    print(f"Number of images in the folder: {number_of_images}")
    metrics = run_model(model, train_folder, test_folder, epochs, lr, features)
    return metrics

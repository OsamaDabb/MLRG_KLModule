from Image_pipeline_functions import *
from features.image_features import *
from data_augmentation.tm_function import *
from binary_pipeline_functions import *

##########################
# RUN FUNCTION
##########################

# This combines the binary and image pipelines and parses the parameters
# def run_function(cancer_list, binary_boolean, image_boolean, 
#         TM_use, value, lin_int, noise, noise_type, 
#         augmentation, shear_factor, shear_prop, crop_scale_factor, crop_scale_prop, 
#         flip_code, flip_prop, rotation_angle, rotate_prop,
#         color_number, color_prop, blur_param, blur_prop,
#         features, glcm_distance, glcm_angle, glcm_prop, lbp_radius, lbp_prop, gabor_prop, haralick_prop,
#         lbp_n_points, gabor_orientations, gabor_scales, gabor_bandwith, Machine, lr, epochs):

# TODO: Make it so that you don't have to specify which train_folder exists. Make a while loop such that while there exists a train folder that exists, you check if train_folder + "_1" exists, and if not use it, else continue. Do same with test folder.
def run_function(cancer_list, binary_boolean, image_boolean, 
        TM_use, value, lin_int, noise, noise_type, 
        augmentation, shear_factor, shear_prop, crop_scale_factor, crop_scale_prop, 
        flip_code, flip_prop, rotation_angle, rotate_prop,
        color_number, color_prop, blur_param, blur_prop,
        features, glcm_distance, glcm_angle, glcm_prop, lbp_radius, lbp_prop, haralick_prop,
        Machine, lr, epochs):

    print("MAIN")
    # If augmentation is true, make a list of the augmentation parameters
    if augmentation:
        augmentation_params = {"rotation_angle": rotation_angle, "flip_code": flip_code, "crop_scale_factor": crop_scale_factor, "shear_factor": shear_factor, "rotate_prop": rotate_prop,
                               "flip_prop": flip_prop, "crop_scale_prop": crop_scale_prop, "color_number": color_number, "blur_param": blur_param, "color_prop": color_prop}
    else:
        augmentation_params = None
    # Similarly, make list of feature parameters if features is true
    if features:
        participation_factors = {'glcm_prop':glcm_prop, 'lbp_prop':lbp_prop, 'haralick_prop':haralick_prop}
        #feature_params = {'glcm': {'distance': glcm_distance, 'angle': glcm_angle}, 'lbp': {'radius': lbp_radius, 'n_points': lbp_n_points}, 
         #                 'gabor': {'orientations': gabor_orientations, 'scales': gabor_scales, 'bandwidth': gabor_bandwith}}
        feature_params = {'glcm': {'distance': glcm_distance, 'angle': glcm_angle}, 'lbp': {'radius': lbp_radius}}

    # Make the return dictionaries empty so we can append later
    params = None
    metrics = None
    # cancer_list should be in the form "Colon,Lung" or something similar
    cancer_list = cancer_list.split(',')

    # For whatever reason booleans weren't working here so I've used strings, w/e
    # Goes into binary pipeline if you chose binary
    if binary_boolean == "True":
        print("BINARY")
        # Goes through each cancer in the cancer list and retrieves the data, phenotype, and enhanced genes (for TM)
        for cancer in cancer_list:
            if cancer.lower() == "binary colon":
                data = colon_data
                phenotype = colon_phenotype
                enhanced_genes = colon_enhanced_genes
            elif cancer.lower() == "binary lung":
                data = lung_data
                phenotype = lung_phenotype
                enhanced_genes = lung_enhanced_genes
            elif cancer.lower() == "binary prostate":
                data = prostate_data
                phenotype = prostate_phenotype
                enhanced_genes = prostate_enhanced_genes
            elif cancer.lower() == "binary_imported":
                folder1 = binary_imported_normal
                folder2 = binary_imported_scc
            else:
                # If cancer type isn't found, raise error
                raise ValueError(f"Unsupported cancer type: {cancer}")

            # This shouldn't really matter but is a safeguard
            if data is None or phenotype is None:
                raise ValueError("You must enter phenotype and RNA sequencing data.")
            else:
                # This was for testing with random binary params, uncomment to select random
                # value, noise_type = select_random_binary_params()

                # model_list should be in the form of "svm,knn" or something similar
                # model_list = model_type.split(',')
                # Runs the binary pipeline for every machine chosen in model_list
                # for model in model_list:
                    # print(model)
                params, metrics = binary_pipeline(data, phenotype, Machine, TM_use, value, enhanced_genes, lin_int,
                                                    noise, noise_type)
                print(f"Dataset {cancer} using machine {Machine} had parameters {params} and metrics {metrics}")
                # TODO: This only returns the last cancer entered right now, have to fix that
                # Fixed I think?

    # Similarly done for image pipeline
    if image_boolean == "True":
        print("IMAGE")
        for cancer in cancer_list:
            if cancer.lower() == "image oral":
                folder1 = oral_normal
                folder2 = oral_scc
            elif cancer.lower() == "image blood":
                folder1 = all_normal
                folder2 = all_scc
            elif cancer.lower() == "image breast":
                folder1 = breast_normal
                folder2 = breast_scc
            elif cancer.lower() == "image colon":
                folder1 = colon_normal
                folder2 = colon_scc
            elif cancer.lower() == "image lung":
                folder1 = lung_normal
                folder2 = lung_scc
            elif cancer.lower() == "image chest x-ray":
                folder1 = chest_normal
                folder2 = chest_scc
            elif cancer.lower() == "image breast ultra sound":
                folder1 = breast_ultra_normal
                folder2 = breast_ultra_scc
            elif cancer.lower() == "image_imported":
                folder1 = image_imported_normal
                folder2 = image_imported_scc
            else:
                raise ValueError(f"Unsupported cancer type: {cancer}")
            
            # Failsafe for the future in case we have inputs for folders
            if folder1 is None or folder2 is None:
                raise ValueError("You must enter Image Data")
            
            # Makes the image dataframe, does augmentation if called in main
            params = pipeline_image_df(folder1, folder2, augmentation, random_params = False, user_params=augmentation_params)
            
            # Runs the features on the image dataframe if called
            # if features:
                # normal_df, scc_df, params = pipeline_features(normal_df, scc_df, participation_factors, params=feature_params)
            
            # Runs the machine pipeline on the dataframes on any given model type
            if augmentation == True:
                metrics = pipeline_machine(folder1 + "_augmented", folder2 + "_augmented", Machine, epochs=epochs, lr=lr, features = features)
            else:
                metrics = pipeline_machine(folder1+ "_temp", folder2 + "_temp", Machine, epochs=epochs, lr=lr, features = features, params = participation_factors)
            if features and augmentation:
                params = {**augmentation_params, **feature_params}
            elif features:
                params = feature_params
            elif augmentation:
                params = augmentation_params
            else:
                params = {"cancer": cancer_list, "machine": Machine, "lr": lr, "epochs": epochs}
            print(f"Dataset {cancer} using machine {Machine} had parameters {params} and metrics {metrics}")
    if TM_use == False:
        return params, metrics
    else:
        DE_csv = return_enhanced_genes(enhanced_genes, value)
        return params, metrics, DE_csv
# run_function(cancer_list = "Colon", binary_boolean = "True", image_boolean = "False", model_type = "svm", 
#         TM_use = False, value = None, lin_int = False, noise = False, noise_type = None, 
#         augmentation = False, features = False, angle = None, flip_code = None, crop_size = None, shear_factor = None, rotate_prop = None, flip_prop = None, 
#         crop_scale_prop = None, color_number = None, kernel_size = None, 
#         glcm_participation = None, lbp_participation = None, gabor_participation = None, haralick_participation = None,
#         glcm_distance = None, glcm_angle = None, lbp_radius = None, lbp_n_points = None, gabor_orientations = None, gabor_scales = None, gabor_bandwith = None)

    


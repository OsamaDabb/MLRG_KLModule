# image_features.py 
class containing image feature functions including GLCM, LBP & Haralick. Given array of images, returns the processed images with a percent of them processed through GLCM, and a percent through LBP. Optionally returns an array of Haralick features for each image.

All interaction should go through Image_Features.generate_image_features():
    params:
    - images: np array or similar of images 
    - percent_glcm: float to calculate what percent of images use GLCM, percent
    lbp is 1 - percent_glcm
    - distance, angle: (floats) hyperparameters for compute_glcm
    -  radius, n_points: (floats) hyperparameters for compute lbp
    - use_haralick: (bool) whether or not to compute haralick features
    - return_mean: (bool) hyperparamter for computer_haralick_features. 

Dependencies:
- mahotas
- skimage

import cv2
import os
import numpy as np
import shutil
import random
import tempfile

class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def rotate(self, angle):
        # Define the allowed angles
        allowed_angles = [0, 90, 180, 270]

        # Check if the provided angle is in the allowed list
        if angle not in allowed_angles:
            raise ValueError("Angle must be one of " + str(allowed_angles))

        # Get the dimensions of the image
        (height, width) = self.image.shape[:2]

        # Compute the center of the image
        center = (width // 2, height // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Adjust the image dimensions if the angle is 90 or 270 degrees
        if angle in [90, 270]:
            new_width, new_height = height, width
        else:
            new_width, new_height = width, height

        # Perform the rotation
        self.image = cv2.warpAffine(self.image, M, (new_width, new_height))
        return self.image

    def flip(self, flip_code):
        if flip_code in [-1, 0, 1]:
            self.image = cv2.flip(self.image, flip_code)
        else:
            raise ValueError("Flip code must be -1 (both), 0 (vertical), or 1 (horizontal)")
        return self.image

    def crop_and_scale(self, crop_size):
        # Define the allowed crop sizes
        allowed_crop_sizes = list(range(50, 501, 50))  # Generates [50, 100, 150, ..., 500]

        # Check if the provided crop size is in the allowed list
        if crop_size not in allowed_crop_sizes:
            raise ValueError("Crop size must be one of " + str(allowed_crop_sizes))

        original_height, original_width = self.image.shape[:2]

        # Ensure the crop size is not larger than the image dimensions
        crop_size = min(crop_size, original_height, original_width)

        # Calculate the starting points for the crop
        start_x = original_width // 2 - crop_size // 2
        start_y = original_height // 2 - crop_size // 2

        # Crop the image
        cropped_image = self.image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
        # Scale the cropped image back to the original size
        self.image = cv2.resize(cropped_image, (original_width, original_height), interpolation=cv2.INTER_AREA)

        return self.image

    def shear(self, shear_factor):
        # Define the allowed shear factors
        # allowed_shear_factors = [0.1 * i for i in range(1, 11)]
        allowed_shear_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Check if the provided shear factor is in the allowed list
        if shear_factor not in allowed_shear_factors:
            raise ValueError("Shear factor must be one of " + str(allowed_shear_factors))

        # Proceed with the shearing operation
        rows, cols, _ = self.image.shape
        M = np.float32([[1, shear_factor, 0],
                        [0, 1, 0]])
        sheared_image = cv2.warpAffine(self.image, M, (cols + int(rows * abs(shear_factor)), rows))

        # Calculate new dimensions
        new_cols = cols
        new_rows = rows

        # Crop or pad the image to maintain original size
        if sheared_image.shape[1] > cols:
            # Crop the image if it's wider than the original
            start_col = int((sheared_image.shape[1] - cols) / 2)
            self.image = sheared_image[:, start_col:start_col + cols]
        else:
            # Pad the image if it's narrower than the original
            padding = (cols - sheared_image.shape[1]) // 2
            self.image = cv2.copyMakeBorder(sheared_image, 0, 0, padding, padding, cv2.BORDER_CONSTANT)

        return self.image


def augment_images_with_proportions(input_folder, output_folder, angle, flip_code, crop_size, shear_factor, rotate_prop, flip_prop, crop_scale_prop):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List of image files
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")]
    random.shuffle(image_files)  # Shuffle the list to randomize selection

    total_images = len(image_files)
    proportions = [rotate_prop, flip_prop, crop_scale_prop, 1 - (rotate_prop + flip_prop + crop_scale_prop)]

    # Calculate the number of images for each method
    images_per_method = [int(prop * total_images) for prop in proportions]

    # Ensure the sum of all images equals the total number of images
    images_per_method[-1] += total_images - sum(images_per_method)  # Adjust the last method count if needed

    methods = ['rotate', 'flip', 'crop_and_scale', 'shear']
    image_index = 0

    for i, method in enumerate(methods):
        for _ in range(images_per_method[i]):
            if image_index < len(image_files):
                filename = image_files[image_index]
                input_path = os.path.join(input_folder, filename)

                # Copy original image to output folder
                shutil.copy(input_path, os.path.join(output_folder, filename))

                # Read and process the image
                image = cv2.imread(input_path)
                processor = ImageProcessor(image)

                if method == 'rotate':
                    augmented_image = processor.rotate(angle)
                elif method == 'flip':
                    augmented_image = processor.flip(flip_code)
                elif method == 'crop_and_scale':
                    augmented_image = processor.crop_and_scale(crop_size)
                else:  # shear
                    augmented_image = processor.shear(shear_factor)

                augmented_filename = f"{filename.split('.')[0]}_{method}.jpg"
                cv2.imwrite(os.path.join(output_folder, augmented_filename), augmented_image)
            
            image_index += 1

class ImageProcessor2:
    def __init__(self, image):
        self.image = image

    def segment_image(self, color_number):
        # Define HSV color ranges for specific numbers
        color_ranges = {
            0: ((0, 100, 100), (10, 255, 255)),     # Red
            1: ((11, 100, 100), (25, 255, 255)),    # Orange
            2: ((26, 100, 100), (34, 255, 255)),    # Yellow
            3: ((35, 100, 100), (85, 255, 255)),    # Green
            4: ((86, 100, 100), (125, 255, 255)),   # Blue
            5: ((126, 100, 100), (145, 255, 255))   # Purple
        }

        # Check if the provided color number is valid
        if color_number not in color_ranges:
            raise ValueError("Color number must be one of " + str(list(color_ranges.keys())))

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Get the lower and upper color range for the specified number
        lower_color, upper_color = color_ranges[color_number]

        # Create a mask and apply it to segment the image
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        self.image = cv2.bitwise_and(self.image, self.image, mask=mask)

        return self.image

    def blur_image(self, kernel_size):
        # Define the allowed kernel sizes
        allowed_kernel_sizes = list(range(3, 22, 2))  # Generates [3, 5, 7, 9, ..., 21]

        # Check if the provided kernel size is in the allowed list
        if kernel_size not in allowed_kernel_sizes:
            raise ValueError("Kernel size must be one of " + str(allowed_kernel_sizes))

        # Proceed with the Gaussian blurring
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        return self.image

def augment_images_with_proportions_2(input_folder, output_folder, color_number, kernel_size, segment_prop):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List of image files
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")]
    random.shuffle(image_files)  # Shuffle the list to randomize selection

    total_images = len(image_files)
    proportions = [segment_prop, 1 - segment_prop]

    # Calculate the number of images for each method
    images_per_method = [int(prop * total_images) for prop in proportions]

    # Ensure the sum of all images equals the total number of images
    images_per_method[-1] += total_images - sum(images_per_method)  # Adjust the last method count if needed

    methods = ['segment', 'blur']
    image_index = 0

    for i, method in enumerate(methods):
        for _ in range(images_per_method[i]):
            if image_index < len(image_files):
                filename = image_files[image_index]
                input_path = os.path.join(input_folder, filename)

                # Copy original image to output folder
                shutil.copy(input_path, os.path.join(output_folder, filename))

                # Read and process the image
                image = cv2.imread(input_path)
                processor = ImageProcessor2(image)

                if method == 'segment':
                    augmented_image = processor.segment_image(color_number)  # Corrected method name
                else:  # blur
                    augmented_image = processor.blur_image(kernel_size)

                augmented_filename = f"{filename.split('.')[0]}_{method}.jpg"
                cv2.imwrite(os.path.join(output_folder, augmented_filename), augmented_image)
        
            image_index += 1

# def combined_augment_function(input_directory, final_directory, 
#                               angle, flip_code, crop_size, shear_factor, 
#                               rotate_prop, flip_prop, crop_scale_prop, 
#                               color_number, kernel_size, segment_prop):

def combined_augment_function(input_directory, final_directory, 
                              rotation_angle, flip_code, crop_scale_factor, shear_factor, 
                              rotate_prop, flip_prop, crop_scale_prop, 
                              color_number, blur_param, color_prop):

    with tempfile.TemporaryDirectory() as intermediate_directory:
        print(f"Intermediate directory: {intermediate_directory}")
        # First augmentation process
        augment_images_with_proportions(input_directory, intermediate_directory, 
                                        rotation_angle, flip_code, crop_scale_factor, shear_factor, 
                                        rotate_prop, flip_prop, crop_scale_prop)
        print("First augmentation process completed.")
        # Second augmentation process
        augment_images_with_proportions_2(intermediate_directory, final_directory, 
                                          color_number, blur_param, color_prop)
        print("Second augmentation process completed.")

# Example usage
# combined_augment_function('C:\\Users\\You\\Desktop\\raw image\\', 'C:\\Users\\You\\Desktop\\final image\\',
#                               rotation_angle=90, flip_code=1, crop_size=100, shear_factor=0.1, 
#                               rotate_prop=0.3, flip_prop=0.3, crop_scale_prop=0.3, 
#                               color_number=5, kernel_size=7, segment_prop=0.3)

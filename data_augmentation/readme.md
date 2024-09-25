# Data Augmentation

6 augmentation mathods, 4 geometry and 2 color.

**rotate**

angle: determines the degree of rotation applied to the image.

Choose from 0, 90, 180, 270.

**flip**

filp_code: specifies how an image should be flipped.

Choose from -1, 0, 1.

-1: Flips the image both vertically and horizontally.

0: Flips the image vertically (around the x-axis).

1: Flips the image horizontally (around the y-axis).

**crop_and_scale**

crop_size:specifies the size of the square area to be cropped from the image. 

Choose from 50, 100, 150, ... , 500.

**shear**

shear_factor: specifies the intensity of the shearing transformation applied to the image.

Choose from 0.1, 0.2, ... , 1.0.

**segment_image**

color_number: selects a specific color range for image segmentation. 

Choose from 0 (red), 1 (orange), 2 (yellow), 3 (green), 4 (blue), 5 (purple).

**blur_image**

kernel_size: refers to the size of the kernel used for the Gaussian blur operation.

Choose from 3, 5, 7, ... , 19, 21.

**proportion**

rotate_prop, flip_prop and crop_scale_prop: the proportions of the total images that will be rotated, flipped, and cropped and scaled, respectively.

the proportion of the total images that will be sheared will be (1 - rotate_prop - flip_prop - crop_scale_prop).

segment_prop: the proportions of the total images that will be segmented.

the proportion of the total images that will be sheared will be (1 - segment_prop).

Waiting tasks: TM function is still waiting. We should talk about that. Next we are still looking for the binary(human genome data set) data augmentation tools in one class.
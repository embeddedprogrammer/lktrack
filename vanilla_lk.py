import numpy as np

from scipy import misc

# for image_path in glob.glob("/home/adam/*.png"):
#     image = misc.imread(image_path)
#     print image.shape
#     print image.dtype
img = misc.imread("imgs/bat2.png")
misc.imshow(img)
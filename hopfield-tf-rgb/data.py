import numpy as np
#import visualize
from PIL import Image




class Dataset:
    def __init__(self, img_file):
        im = Image.open(img_file)
        im = im.convert("RGB")

        self.image_array = np.array([a[0:3] for a in im.getdata()])
        self.pixel_count = self.image_array.shape[0]
        self.image_size = im.size

        self.image_matrix = np.reshape(self.image_array, newshape=(im.size[0], im.size[1], 3)) / 255.0
        self.image_matrix_grayscale = np.average(self.image_matrix, axis=2)

        print("Loaded image of size {}".format(im.size))


    def get_image_matrix_grayscale(self):
        return self.image_matrix_grayscale

    def get_image_matrix_rgb(self):
        return self.image_matrix





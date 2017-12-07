import numpy as np
from PIL import Image
import os
from image import *


filename = "./test.jpg"
image_size = (480, 280)

w, h = image_size
image_inputs = image2inputs(filename, w, h)

def get_random_pattern():
    # the following lines can be optimized using numpy
    i = np.random.randint(len(image_inputs))
    input = image_inputs[i]

    return input

def get_all_patterns():
    return image_inputs


if __name__ == "__main__":
    i = get_random_pattern()
    print "get_random_pattern:", i


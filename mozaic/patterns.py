import numpy as np
from PIL import Image
import os
from image import *
import test

tile_files = ["./tile"+str(a)+".jpg" for a in [1,2,3,4,5,6,7,8]]
filename = "./test.jpg"
image_size = (480, 280)

choose_standard_tile_prob = 0.6


w, h = image_size
image_inputs = image2inputs(filename, w, h)
tiles_inputs = [image2inputs(tfile, 10, 10)[0] for tfile in tile_files]

def get_random_pattern():
    # the following lines can be optimized using numpy
    if np.random.randint(1e6) < choose_standard_tile_prob * 1e6:
        input = tiles_inputs[np.random.randint(len(tiles_inputs))]
    else:
        input = image_inputs[np.random.randint(len(image_inputs))]

    return input

def get_all_patterns():
    return image_inputs

def is_standard_tile(input):
    for standard in tiles_inputs:
        if test.get_diff(standard, input) == 0:
            return True


    return False


if __name__ == "__main__":
    i = get_random_pattern()
    print "get_random_pattern:", i


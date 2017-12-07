import numpy as np
from PIL import Image
import os

tile_w = 10
tile_h = 10


def _image2array(path):
    im = Image.open(path)
    arr = list(im.getdata())
    arr = [(a + b + c) / 3 / 255.0 for a, b, c in arr]
    return arr

def _round_inputs(inputs):
    return [[0 if a < 0.5 else 1 for a in input] for input in inputs]

def inputs2image(inputs, width, height, round = True):
    if round:
        inputs = _round_inputs(inputs)

    img_data = np.zeros((height, width, 3), dtype=np.uint8)


    inputs_i = 0
    for ii in xrange(0, height, tile_h):
        for i in xrange(0, width, tile_w):
            inputs_ii = 0
            for y in range(ii, ii + tile_h):
                for x in range(i, i + tile_w):
                    val = inputs[inputs_i][inputs_ii] * 255
                    img_data[y, x] = [val,val,val]
                    inputs_ii += 1
            inputs_i += 1

    return Image.fromarray(img_data)


def get_input_type_count(inputs):
    inputs = _round_inputs(inputs)
    arr = []

    for input in inputs:
        hash = str(input)
        # or better?

        if not hash in arr:
            arr.append(hash)

    return len(arr)

def image2inputs(path, width, height):
    arr = _image2array(path)

    ret = []
    for ii in xrange(0, height, tile_h):
        #ret_row = []
        for i in xrange(0, width, tile_w):
            ret_item = []
            for y in range(ii, ii + tile_h):
                for x in range(i, i + tile_w):
                    ret_item.append(arr[y*width + x])
            #ret_row.append(ret_item)
            ret.append(ret_item)
        #ret.append(ret_row)

    return ret


if __name__ == "__main__":
    #print "Pattern"
    path = "test.jpg"
    inputs = image2inputs(path, 480, 280)
    print "image2inputs:", inputs
    image = inputs2image(inputs)
    print "inputs2image (the image should be sama as original is working well:"
    # TODO display image

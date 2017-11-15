import numpy as np
from PIL import Image
import os

# using 6x14 images

chars = ['/', '\\', '|', '-', '_', '^', ' ']

space_treshold = 0.5

def get_pattern():
    c = np.random.randint(len(chars))
    input = chari2randin(c)
    output = chari2out(c)

    return input, output


def chari2randin(c):
    dir = "imgs/{0}/".format(c)
    files = os.listdir(dir)
    file = files[np.random.randint(len(files))]

    ret = image_as_array(dir+file)

    return ret

def chari2out(c):
    if c == 0:
        return [1, 0, 0, 0, 0, 0]
    if c == 1:
        return [0, 1, 0, 0, 0, 0]
    if c == 2:
        return [0, 0, 1, 0, 0, 0]
    if c == 3:
        return [0, 0, 0, 1, 0, 0]
    if c == 4:
        return [0, 0, 0, 0, 1, 0]
    if c == 5:
        return [0, 0, 0, 0, 0, 1]
    if c == 6:
        return [0, 0, 0, 0, 0, 0]
    raise IndexError

def out2char(output):
    i = 0
    maxI = 0
    max = 0
    space = True

    for a in output:
        if a > max:
            max = a
            maxI = i

        if a > space_treshold:
            space = False

        i += 1

    if space:
        return " "
    else:
        return chars[maxI]

def image_as_array(path):
    im = Image.open(path)
    arr = list(im.getdata())
    arr = [(a + b + c)/3/255.0 for a,b,c in arr]
    return arr


def get_outputs_from_image(path, width, height):
    arr = image_as_array(path)

    ret = []
    for ii in xrange(0, height, 14):
        ret_row = []
        for i in xrange(0, width, 6):
            ret_item = []
            for y in range(ii, ii + 14):
                for x in range(i, i + 6):
                    ret_item.append(arr[y*width + x])
            ret_row.append(ret_item)
        ret.append(ret_row)
    return ret

if __name__ == "__main__":
    print "Pattern"
    i, o = get_pattern()
    print "Input:", i
    print "Out:", o

    print "Outputs"
    print get_outputs_from_image("imgs/test.bmp")
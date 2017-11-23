import numpy as np
from PIL import Image
import os
import shutil

# using 6x14 images



def open_source(src, rel_path):
    im = Image.open(src + rel_path + ".bmp")
    return im

def save_to(dst, dirname, im):
    name = str(np.random.randint(10000000)) + ".bmp"
    # os.exi
    dir = dst + dirname

    if not os.path.exists(dir):
        os.mkdir(dir)

    im.save(dir + "/" + name)

def clear_dst(dst):
    #os.remove(dst)
    if os.path.exists(dst):
        shutil.rmtree(dst)

    os.mkdir(dst)

def preprocess(src, dst):
    clear_dst(dst)
    im = open_source(src, "0/1")

    im2 = move(im,4,4)

    #save_to(dst, "0", im)
    im.show()
    im2.show()

def move(im, left, top):
    temp = im.copy()
    w, h = im.size


    for x in range(w):
        for y in range(h):
            temp.putpixel((x, y), 0)


    for x in range(w):
        for y in range(h):
            if x + left < w and y + top < h and x + left >= 0 and y + top >= 0:
                pix = im.getpixel((x, y))
                temp.putpixel((x + left, y + top), pix)



    return temp


if __name__ == "__main__":

    preprocess("imgs/src/", "imgs/preprocessed/")
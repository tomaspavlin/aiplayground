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

    # / and \
    dst_rel = "0"
    flip = False
    for i in range(2):

        preprocess_image(src, "0/1", dst, dst_rel, 0, 0, 0, 8, flip)
        preprocess_image(src, "0/2", dst, dst_rel, 0, 0, 0, 7, flip)
        preprocess_image(src, "0/3", dst, dst_rel, 0, 0, 0, 6, flip)
        preprocess_image(src, "0/4", dst, dst_rel, 0, 0, 0, 5, flip)
        preprocess_image(src, "0/5", dst, dst_rel, 0, 0, 0, 4, flip)
        preprocess_image(src, "0/6", dst, dst_rel, 0, 0, 0, 3, flip)
        preprocess_image(src, "0/7", dst, dst_rel, 0, 0, 0, 2, flip)
        preprocess_image(src, "0/8", dst, dst_rel, 0, 0, 0, 1, flip)
        preprocess_image(src, "0/9", dst, dst_rel, 0, 0, 0, 0, flip)

        dst_rel = "1"
        flip = True

    # |
    preprocess_image(src, "2/1", dst, "2", 0, 0, 5, 0)
    preprocess_image(src, "2/2", dst, "2", 0, 0, 4, 0)
    preprocess_image(src, "2/3", dst, "2", 0, 0, 3, 0)
    preprocess_image(src, "2/2", dst, "2", 0, 0, 4, 0, True)
    preprocess_image(src, "2/3", dst, "2", 0, 0, 3, 0, True)

    # -
    preprocess_image(src, "3/1", dst, "3", 0, 0, 0, 9)

    # _
    preprocess_image(src, "3/1", dst, "4", 0, 10, 0, 13)

    # ^
    preprocess_image(src, "5/1", dst, "5", 0, 0, 0, 3)
    preprocess_image(src, "5/1", dst, "5", 0, 0, 0, 3, True)

    # space
    preprocess_image(src, "6/1", dst, "6", 0, 0, 0, 0)


def preprocess_image(src, rel, dst, dst_rel, min_x, min_y, max_x, max_y, flip = False):

    im = open_source(src, rel)

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            im2 = move(im, x, y, flip)
            save_to(dst, dst_rel, im2)
            print "Flipped image" if flip else "Image", rel, "moved by",x,"and",y,"pixels", "and saved to", dst_rel




def move(im, left, top, flip):
    temp = im.copy()
    w, h = im.size


    for x in range(w):
        for y in range(h):
            temp.putpixel((x, y), (255, 255, 255))


    for x in range(w):
        for y in range(h):
            if x + left < w and y + top < h and x + left >= 0 and y + top >= 0:
                pix = im.getpixel((x, y))
                if flip:
                    temp.putpixel((w - (x + left) - 1, h - (y + top) - 1), pix)
                else:
                    temp.putpixel((x + left, y + top), pix)



    return temp


if __name__ == "__main__":

    preprocess("imgs/src/", "imgs/preprocessed/")
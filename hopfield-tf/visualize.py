import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFont

pallete_square_size = 20
background = (100, 255, 100)
linecolor = (0,0,0)
linewidth = 1
nodecolor = (255,0,0)
noderadius_grid = 1.5
noderadius_dot = 0.5
textcolor = (0, 0, 0)
zoom = 0.6
max_xy = 400
size = (int(zoom*max_xy), int(zoom*max_xy))
axis_step = 10


def visualize_grid(grid, dots, filename, text=""):

    im = Image.new('RGB', size, background)
    draw = ImageDraw.Draw(im)

    # add axis
    for i in range(3):
        for ii in range(0, 256, axis_step):

            for iii in range(3):
                cc = [0, 0, 0]
                if iii == i:
                    cc[i] = ii
                else:
                    cc[i] = 255
                    cc[iii] = ii

                dots.append(cc)

    _drawdots(draw, dots, 1, noderadius_dot)

    _drawlines(draw, grid)
    _drawdots(draw, grid, 2, noderadius_grid)

    draw.text((0, 0), text, fill=textcolor)

    im.show()
    im.save(filename)


def visualize_palette(grid, filename):
    size = [a*pallete_square_size for a in grid.shape[:2]]

    im = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(im)

    _drawsquares(draw, grid, pallete_square_size)

    #im.show()
    im.save(filename)

def visualize_image_matrix_grayscale(image_matrix, filename):
    flattened = np.reshape(image_matrix, newshape=(image_matrix.shape[0]*image_matrix.shape[1]))


    im = Image.new('RGB', image_matrix.shape)

    pixels = [(int(a * 255), int(a * 255), int(a * 255)) for a in flattened]
    im.putdata(pixels)

    im.save(filename)

def visualize_image(image_data, image_size, filename, matrix):
    im = Image.new('RGB', image_size)


    data = []
    i = 0
    for i, pixel in enumerate(image_data):

        pixel, _, _ = kohonen.get_nearest_neuron(pixel, matrix)

        pixel = [int(a) for a in pixel]

        data.append(tuple(pixel))

        if not i % 10000:
            print(100*i/len(image_data))

    im.putdata(data)

    im.save(filename)

def copy_image(from_file, to_file):
    im = Image.open(from_file)
    im.save(to_file)

def clear_image(filename):
    im = Image.open(filename)

    im2 = Image.new('RGB', im.size, background)
    im2.save(filename)



def _drawlines(draw, grid):
    for iy in range(len(grid)):
        for ix in range(len(grid[iy])):
            p = _project3Dto2d(grid[iy][ix])

            if iy > 0:
                draw.line((p[0]*zoom, p[1]*zoom,
                           _project3Dto2d(grid[iy-1][ix])[0]*zoom,
                           _project3Dto2d(grid[iy-1][ix])[1]*zoom), fill=linecolor, width=linewidth)

            if ix > 0:
                draw.line((p[0]*zoom, p[1]*zoom,
                           _project3Dto2d(grid[iy][ix-1])[0]*zoom,
                           _project3Dto2d(grid[iy][ix-1])[1]*zoom), fill=linecolor, width=linewidth)

def _drawdots(draw, dots, dim, noderadius):
    if dim > 0:
        for dots2 in dots:
            _drawdots(draw, dots2, dim - 1, noderadius)
    else:
        p = _project3Dto2d(dots)
        color = tuple(int(a) for a in dots)


        draw.ellipse((p[0] * zoom - noderadius, p[1] * zoom - noderadius,
                      p[0] * zoom + noderadius, p[1] * zoom + noderadius), fill=color)

def _drawsquares(draw, grid, squaresize):
    for iy in range(len(grid)):
        for ix in range(len(grid[iy])):
            p = (iy * squaresize, ix * squaresize)

            color = tuple(int(a) for a in grid[iy][ix])


            draw.rectangle((p[0], p[1],
                p[0] + squaresize, p[1] + squaresize),
                fill=color)

def _project3Dto2d(p):


    return p[0] + p[2]//2, p[1] + p[2]//3


if __name__ == "__main__":
    arr = [
        [
            (10, 10),
            (20, 10),
            (30, 10)
        ],
        [
            (10, 40),
            (20, 40),
            (30, 40)
        ]
    ]

    dots = [
        (20, 20),
        (30, 30),
        (40, 40),
        (50, 50)
    ]

    text = "Hello"

    visualize(arr, dots, text)
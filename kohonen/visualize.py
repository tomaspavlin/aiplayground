import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageDraw
import webbrowser
import os


linecolor = (0,0,0)
linewidth = 3
nodecolor = (255,0,0)
noderadius = 3
zoom = 3


def visualize(grid, dots):
    im = Image.new('RGB', (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    _drawlines(draw, grid)
    # _drawdots(draw, grid, 2)
    _drawdots(draw, dots, 1)

    im.show()
    im.save("image.png")
    print("Image saved")

def _drawlines(draw, grid):
    for iy in range(len(grid)):
        for ix in range(len(grid[iy])):
            p = grid[iy][ix]

            if iy > 0:
                draw.line((p[0]*zoom, p[1]*zoom,
                           grid[iy-1][ix][0]*zoom, grid[iy-1][ix][1]*zoom), fill=linecolor, width=linewidth)

            if ix > 0:
                draw.line((p[0]*zoom, p[1]*zoom,
                           grid[iy][ix-1][0]*zoom, grid[iy][ix-1][1]*zoom), fill=linecolor, width=linewidth)

def _drawdots(draw, dots, dim):
    if dim > 0:
        for dots2 in dots:
            _drawdots(draw, dots2, dim - 1)
    else:
        p = dots
        draw.ellipse((p[0] * zoom - noderadius, p[1] * zoom - noderadius,
                      p[0] * zoom + noderadius, p[1] * zoom + noderadius), fill=nodecolor)



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

    visualize(arr, dots)
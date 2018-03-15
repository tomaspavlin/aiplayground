import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

background = (200, 200, 200)
linecolor = (0,0,0)
linewidth = 3
nodecolor = (255,0,0)
noderadius = 3
textcolor = (0, 0, 0)
zoom = 60
max_xy = 10
size = (zoom*max_xy, zoom*max_xy)


def visualize(grid, dots, text=""):
    im = Image.new('RGB', size, background)
    draw = ImageDraw.Draw(im)

    _drawdots(draw, dots, 1)

    #_drawlines(draw, grid)
    #_drawdots(draw, grid, 2)

    draw.text((0, 0), text, fill=textcolor)

    im.show()
    im.save("image.png")
    #print("Image saved")

def _drawlines(draw, grid):
    for iy in range(len(grid)):
        for ix in range(len(grid[iy])):
            p = _project3Dto2d(grid[iy][ix])

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
        p = _project3Dto2d(dots)
        color = dots

        draw.ellipse((p[0] * zoom - noderadius, p[1] * zoom - noderadius,
                      p[0] * zoom + noderadius, p[1] * zoom + noderadius), fill=color)

def _project3Dto2d(p):
    return p[0] + p[2]//2, p[1] + int(p[2]*0.866)

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
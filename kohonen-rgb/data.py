import numpy as np
import visualize
import kohonen
from PIL import Image

image_array = None

class Dataset:
    def __init__(self, img_file):
        im = Image.open(img_file)
        self.image_array = np.array(list(im.getdata()))

    def next_batch(self, n):
        return np.random.choice(self.image_array, n)

    def get_all(self):
        return self.image_array
import numpy as np
import visualize
import kohonen
from PIL import Image

#image_array = None

class Dataset:
    def __init__(self, img_file):
        im = Image.open(img_file)
        im = im.convert("RGB")

        self.image_array = np.array([a[0:3] for a in im.getdata()])
        self.pixel_count = self.image_array.shape[0]
        self.image_size = im.size

    def next_batch(self, n):
        ids = np.random.randint(self.pixel_count, size=n)
        return self.image_array[ids]

    def get_all(self):
        return self.image_array

    def get_every_nth(self, n):
        return self.image_array[[a*2 for a in range(self.pixel_count/n)]]



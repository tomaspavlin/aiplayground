import numpy as np
import visualize
import kohonen
from PIL import Image

#image_array = None

CUBE_N = 4
DATA_N_TRESHOLD = 300

WHAT_CUBE_LEN = 10000



class Dataset:
    def __init__(self, img_file):
        im = Image.open(img_file)
        im = im.convert("RGB")

        self.image_array = np.array([a[0:3] for a in im.getdata()])
        self.pixel_count = self.image_array.shape[0]
        self.image_size = im.size

        self._precompute_cubes()

    def _precompute_cubes(self):

        self.cubes = []
        for _ in range(CUBE_N ** 3):
            self.cubes.append({'n': 0, 'samples': []})


        data_count = len(self.get_all())

        # compute number of samples in each cube
        for sample in self.get_all():
            i = self._get_cube_index(sample)
            self.cubes[i]['n'] += 1

            # add sample to cube
            #self.cubes[i]['samples'].append(sample)


        for i in range(len(self.cubes)):
            n = self.cubes[i]['n']

            # compute probability of sample accept for each cube
            if n > DATA_N_TRESHOLD:
                self.cubes[i]['accept_p'] = float(DATA_N_TRESHOLD) / n
            else:
                self.cubes[i]['accept_p'] = 1

            # compute cubes prob
            self.cubes[i]['p'] = float(n) / data_count

        # precompute for cube search
        #self.what_cube = []
        #p_sum = 0
        #for i in range(WHAT_CUBE_LEN):
        #    if p_sum * WHAT_CUBE_LEN



        # print cubes
        print "Cubes:"
        for i, c in sorted(enumerate(self.cubes), key=lambda a: a[1]['n']):
            if c['n'] != 0:
                print i, c

    def _get_rand_cube(self):
        #r = np.random.randint(WHAT_CUBE_LEN)
        pass





    @staticmethod
    def _get_cube_index(sample):
        #print sample
        cube_size = 256.0/CUBE_N

        ret = 0
        ret += int(sample[0]/cube_size) * CUBE_N
        #print ret
        ret += int(sample[1]/cube_size) * CUBE_N
        #print ret
        ret += int(sample[2]/cube_size)

        return ret

    def next_batch(self, n):
        return self.next_batch_smart(n)

    def next_batch_smart(self, n):
        #return self.next_batch_uniform(n)

        # SMART
        ret = []
        for i in range(n):
            while True:
                sample = self.next_batch_uniform(1)[0]
                accept_prob = self.cubes[Dataset._get_cube_index(sample)]['accept_p']
                if np.random.rand() <= accept_prob:
                    break

            ret.append(sample)

        return ret

    def next_batch_uniform(self, n):
        ids = np.random.randint(self.pixel_count, size=n)
        return self.image_array[ids]

    def get_all(self):
        return self.image_array

    def get_every_nth(self, n):
        return self.image_array[[a*2 for a in range(self.pixel_count/n)]]



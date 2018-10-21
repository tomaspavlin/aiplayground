import numpy as np
import math
import tensorflow as tf



class Network(object):

    def __init__(self):
        self.sess = tf.Session()
        pass

    def construct(self, alpha, image_matrix):
        self.alpha = tf.constant(alpha, dtype=tf.float32, name="alpha")
        self.strmost = tf.placeholder(tf.float32, (), "strmost")
        self.c = tf.constant(image_matrix, dtype=tf.float32, shape=(None, None))

        #w, h = image_matrix.shape

        # construct D
        self._D = self._construct_D()
        self._D = tf.Print(self._D, "Computing D")

        # construct T
        self._T = self._construct_T(self._D)
        self._T = tf.Print(self._T, "Computing T")

        # construct I
        self.I = self._construct_I(self.c)
        self.I = tf.Print(self.I, "Computing I")

        # initialize V
        self.V = self._construct_V(self.c)
        self.V = tf.Print(self.V, "Initializing V")

    def _construct_D(self):
        # _D
        _D = np.ndarray(shape=(9, 9), dtype=np.float32)

        for a in range(9):
            for b in range(9):
                _D[a, b] = (5 - abs(a - 4)) * (5 - abs(b - 4))

        _D = tf.constant(_D, dtype=tf.float32, name="D")

        # D

        return


    def recompute_random(self):
        # get random position
        i, j = np.random.randint(self.c.shape[0] - 8) + 4, np.random.randint(self.c.shape[1] - 8) + 4 # is working?
        #print("GOT pixel {}, {}".format(i, j))

        # compute new

        # g is sigmoid
        inner = self.I[i, j]
        for k in range(i - 4, i + 4 + 1):
            for l in range(j - 4, j + 4 + 1):
                inner += self.T(i, j, k, l) * self.V[k, l] #+ self.I[i, j]

        self.V[i, j] = self._g(inner)


    def compute_cube(self, a):
        ret = self.sess.run(self.cube, {self.a: a})
        return ret


if __name__ == "__main__":
    net = Network()
    net.construct()
    cube = net.compute_cube(5)
    print(f"cube of 5 = {cube}")
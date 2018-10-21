import numpy as np
import math



class Hopfield(object):

    def __init__(self, image_matrix, alpha = 1, strmost = 1):
        """

        :param image_matrix: grayscale, each item is number 0-1

        """

        self.alpha = alpha
        self.strmost = strmost


        self.c = image_matrix

        # compute D
        print("Computing D")
        self._D = Hopfield._compute_D()

        # compute T
        print("Computing T")
        self._T = self._compute_T(self._D)

        # compute I
        print("Computing I")
        self.I = self._compute_I(self.c)

        # initialize V
        print("Initializing V")
        self.V = Hopfield._initialize_V(self.c)

        print("INIT IS DONE :)")


    @staticmethod
    def _compute_D():
        D = np.ndarray(shape=(9, 9), dtype=np.float32)

        for a in range(9):
            for b in range(9):
                D[a, b] = (5 - abs(a - 4)) * (5 - abs(b - 4))

        return D

    def D(self, i, j, k, l):
        # this method is never used
        a = i - k
        b = j - l

        if -5 < a and a < 5 and -5 < b and b < 5:
            return self._D[a + 4, b + 4]
        else:
            return 0

    def _compute_T(self, D):

        T = np.ndarray(shape=(9, 9), dtype=np.float32)
        # maybe np.arange could be used
        for a in range(9):
            for b in range(9):
                TL = 0
                TG = -2 * D[a, b] # a != b ??

                T[a, b] = self.alpha * TL + (1 - self.alpha) * TG

        T[4, 4] = 0 # ...

        return T

    def T(self, i, j, k, l):
        a = i - k
        b = j - l

        if -5 < a and a < 5 and -5 < b and b < 5:
            return self._T[a + 4, b + 4]
        else:
            return 0

    def _compute_I(self, c):
        w, h = c.shape

        I = np.ndarray(shape=c.shape, dtype=np.float32)
        # maybe np.arange could be used
        for i in range(w):
            for j in range(h):
                IL = 2*c[i, j] - 1

                IG = - 25
                for k in range(max(i-4, 0), min(i + 4 +1, w)):
                    for l in range(max(j - 4, 0), min(j + 4 + 1, h)):
                        IG += 2 * self.D(i, j, k, l) * c[k, l]

                I[i, j] = self.alpha * IL + (1 - self.alpha) * IG

        return I

    @staticmethod
    def _initialize_V(c):
        w, h = c.shape

        #V = np.zeros(shape=c.shape, dtype=np.float32)

        V = np.array(c)

        # maybe set to the same values as c

        return V

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

    def _g(self, x):
        return 1 / (1 + np.exp(-x * self.strmost))


    def get_recomputed_image_matrix(self):
        return self.V




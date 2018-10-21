import numpy as np
import math
import time



class Hopfield(object):

    def __init__(self, image_matrix, alpha = 1, strmost = 1):
        """

        :param image_matrix: grayscale, each item is number 0-1

        """

        self.alpha = alpha
        self.strmost = strmost


        self.c = image_matrix
        w, h = self.c.shape

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
        now = time.time()

        print(f"Time: {time.time() - now:.2f}")

        w, h = c.shape

        I = np.ndarray(shape=c.shape, dtype=np.float32)

        ijs = []
        for i in range(w):
            for j in range(h):
                ijs.append((i, j))



        IG_matrix = np.ndarray(shape=c.shape, dtype=np.float32)

        IG_D_matrix = np.ndarray(shape=(c.shape[0], c.shape[1], 9, 9))
        IG_c_matrix = np.ndarray(shape=(c.shape[0], c.shape[1], 9, 9))
        IG_Dc_matrix = np.ndarray(shape=(c.shape[0], c.shape[1], 9, 9))

        if False:
            for i, j in ijs:
                for k in range(max(i - 4, 0), min(i + 4 + 1, w)):
                    for l in range(max(j - 4, 0), min(j + 4 + 1, h)):
                        aa = k - i + 4
                        bb = l - j + 4

                        d = self._D[aa, bb]
                        IG_D_matrix[i, j, aa, bb] = d

        IG_D_matrix = np.tile(self._D, (w, h, 1, 1))

        print(f"Time: {time.time() - now:.2f}")

        def _get_c(i, j, aa, bb):
            k = i + aa - 4
            l = j + bb - 4
            return c[k % w, l % h]

        for i, j in ijs:
            #for aa in range(9):
            #    for bb in range(9):
            for aa, k in enumerate(range(max(i - 4, 0), min(i + 4 + 1, w))):
                for bb, l in enumerate(range(max(j - 4, 0), min(j + 4 + 1, h))):
                    #k = i + aa - 4
                    #l = j + bb - 4
                    IG_c_matrix[i, j, aa, bb] = c[k, l]

        #IG_c_matrix = np
        print(f"Time: {time.time() - now:.2f}")

        if False:
            for i, j in ijs:
                for k in range(max(i - 4, 0), min(i + 4 + 1, w)):
                    for l in range(max(j - 4, 0), min(j + 4 + 1, h)):
                        aa = k - i + 4
                        bb = l - j + 4

                        IG_Dc_matrix[i, j, aa, bb] = IG_D_matrix[i, j, aa, bb] * IG_c_matrix[i, j, aa, bb]

        IG_Dc_matrix = IG_D_matrix * IG_c_matrix

        print(f"Time: {time.time() - now:.2f}")

        if False:
            for i, j in ijs:
                IG = 0
                for k in range(max(i - 4, 0), min(i + 4 + 1, w)):
                    for l in range(max(j - 4, 0), min(j + 4 + 1, h)):
                        aa = k - i + 4
                        bb = l - j + 4

                        IG += IG_Dc_matrix[i, j, aa, bb]

                IG_matrix[i, j] = IG

        IG_matrix = np.sum(np.sum(IG_Dc_matrix, axis=3), axis=2)

        IG_matrix *= 2
        IG_matrix -= 25

        print(f"Time: {time.time() - now:.2f}")

        for i, j in ijs:
            IL = 2*c[i, j] - 1

            #IG = - 25
            #for k in range(max(i-4, 0), min(i + 4 + 1, w)):
            #    for l in range(max(j - 4, 0), min(j + 4 + 1, h)):
            #        IG += 2 * self.D(i, j, k, l) * c[k, l]
            IG = IG_matrix[i, j]

            I[i, j] = self.alpha * IL + (1 - self.alpha) * IG

        print(f"Time: {time.time() - now:.2f}")

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
        i, j = np.random.randint(self.c.shape[0] - 8) + 4, np.random.randint(self.c.shape[1] - 8) + 4  # is working?
        # print("GOT pixel {}, {}".format(i, j))

        # compute new

        # g is sigmoid
        inner = self.I[i, j]
        for k in range(i - 4, i + 4 + 1):
            for l in range(j - 4, j + 4 + 1):
                inner += self.T(i, j, k, l) * self.V[k, l]  # + self.I[i, j]

        self.V[i, j] = self._g(inner)


    def recompute_random_batch(self, n):
        for _ in range(n):
            self.recompute_random()
        return

        # TODO: problem with fading to grayJ


        ijs = np.random.randint(4, self.c.shape[0] - 4, (n, 2))
        for batch in range(n):
            # get random position
            i, j = np.random.randint(self.c.shape[0] - 8) + 4, np.random.randint(self.c.shape[1] - 8) + 4 # is working?
            #print("GOT pixel {}, {}".format(i, j))

            # compute new
            T_matrix = np.ndarray((9, 9))
            V_matrix = np.ndarray((9, 9))

            for a,k in enumerate(range(i - 4, i + 4 + 1)):
                for b,l in enumerate(range(j - 4, j + 4 + 1)):
                    T_matrix[a,b] = self.T(i, j, k, l)
                    V_matrix[a,b] = self.V[k, l]
                    #pass

            TV_matrix = T_matrix * V_matrix

            inner = self.I[i, j] + np.sum(TV_matrix, axis=(0, 1))

            if False:
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




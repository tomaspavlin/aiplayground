import numpy as np
import math
import time



class Hopfield(object):

    def __init__(self, image_matrix, b, alpha, beta, strmost = 1):
        """

        :param image_matrix: rgb, shape: (width, height, 3)

        """

        self.b = np.array(b, dtype=np.float32)

        #print(f"b shape: {self.b.shape}")

        self.alpha = alpha
        self.beta = beta
        self.gamma = 1 - alpha - beta

        #if self.gamma < 0:
        #    raise ValueError("Gamma is negative")
        assert self.gamma >= 0

        self.strmost = strmost


        self.c = image_matrix
        #print(self.c)

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
        self.V = self._initialize_V()

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

        # indexes are (i - i', j - j', b, b)
        T = np.ndarray(shape=(9, 9, self.b.shape[0], self.b.shape[0]), dtype=np.float32)
        print(f"_T shape: {T.shape}")

        # maybe np.arange could be used
        for a in range(T.shape[0]):
            for b in range(T.shape[1]):
                for k in range(T.shape[2]):
                    for l in range(T.shape[3]):
                        if (a, b) == (4, 4) and k != l:
                            TP = -2
                            TL = -2 * sum([self.b[k, t] * self.b[l, t] for t in range(3)])

                        else:
                            TP = 0
                            TL = 0

                        #assert TP == 0
                        T[a, b, k, l] = self.alpha * TP + (1 - self.alpha) * TL

        #T[4, 4] = 0 # ...

        return T

    def T(self, i1, j1, k1, i2, j2, k2):
        a = i1 - i2
        b = j1 - j2

        if -5 < a and a < 6 and -5 < b and b < 6:
            ret = self._T[a + 4, b + 4, k1, k2]
            #assert ret in [0, -2]
            #print(f"a:{a} b:{b} k1:{k1} k2:{k2} ret:{ret}")
            return ret
        else:
            return 0

    def _compute_I(self, c):
        now = time.time()

        print(f"Time: {time.time() - now:.2f}")

        #w, h = c.shape

        I = np.ndarray(shape=[c.shape[0], c.shape[1], self.b.shape[0]], dtype=np.float32)
        IP = np.ones(shape=I.shape, dtype=np.float32)

        # IL
        IL = np.ndarray(shape=I.shape, dtype=np.float32)

        sp = list(I.shape) + [3]
        ILT_c = np.tile(np.reshape(c, (sp[0], sp[1], 1, sp[3])), (1, 1, sp[2], 1))
        ILT_b = np.tile(np.reshape(self.b, (1, 1, sp[2], sp[3])), (sp[0], sp[1], 1, 1))
        ILT = 2 * ILT_c * ILT_b - ILT_b ** 2
        IL = np.sum(ILT, 3)

        if False:
            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    for k in range(I.shape[2]):
                        IL[i, j, k] = sum([2*c[i, j, t]*self.b[k, t] - (self.b[k, t]*self.b[k, t]) for t in range(3)])
                        #IG = - 25
                        #for k in range(max(i - 4, 0), min(i + 4 + 1, w)):
                        #    for l in range(max(j - 4, 0), min(j + 4 + 1, h)):
                        #        IG += 2 * self.D(i, j, k, l) * c[k, l]

                        #I[i, j, k] = self.alpha * IP + (1 - self.alpha) * IL


        I = self.alpha * IP + (1 - self.alpha) * IL
        print("IP correct :)" if int(np.sum(IP, (0, 1, 2))) == 1382400 else "IP INCORRECT ({}) !!!".format(np.sum(IP, (0, 1, 2))))
        print("IL correct :)" if int(np.sum(IL, (0, 1, 2))) == 1658283 else "IL INCORRECT ({}) !!!".format(np.sum(IL, (0, 1, 2))))

        print(f"Time: {time.time() - now:.2f}")

        return I

    def _initialize_V(self):
        #w, h = c.shape

        # maybe random
        #V = np.zeros(shape=[self.c.shape[0], self.c.shape[1], self.b.shape[0]], dtype=np.float32)
        V = np.random.random_sample([self.c.shape[0], self.c.shape[1], self.b.shape[0]])

        #V = np.array()

        # maybe set to the same values as c

        return V

    def recompute_random(self):
        # get random position
        i, j, k = np.random.randint(self.c.shape[0] - 8) + 4, np.random.randint(self.c.shape[1] - 8) + 4, \
            np.random.randint(self.b.shape[0])

        #print("GOT neuron {}, {}, {}".format(i, j, k))

        # compute new

        # g is sigmoid
        #inner = self.I[i, j]
        #for k in range(i - 4, i + 4 + 1):
        #    for l in range(j - 4, j + 4 + 1):
        #        inner += self.T(i, j, k, l) * self.V[k, l]  # + self.I[i, j]

        inner = self.I[i, j, k]
        #print(f"inner: {inner}")
        i2 = i
        j2 = j
        #for i2 in range(i - 4, i + 5):
        #    for j2 in range(j - 4, j + 5):
        for k2 in range(self.b.shape[0]):
            inner += self.T(i, j, k, i2, j2, k2) * self.V[i2, j2, k2]

            #print(self.T(i, j, k, i2, j2, k2), self.V[i2, j2])
            #print(f"inner: {inner}")

        new_V = self._g(inner)
        #print(f"new V: {new_V}")
        self.V[i, j, k] = new_V


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
        indexes = np.argmax(self.V, axis=2)
        ret = self.b[indexes]
        return ret

    def get_recomputed_image_matrix_one_color(self, color_index):
        color = self.b[color_index]
        white = [1, 1, 1]
        w = np.tile(np.reshape(self.V[:, :, color_index], (self.V.shape[0], self.V.shape[1], 1)), (1, 1, 3))
        c = np.tile(np.reshape(white, (1, 1, -1)), (self.V.shape[0], self.V.shape[1], 1))
        ret = w * c
        ret[0:5, :, :] = np.tile(color, (5, ret.shape[0], 1))
        ret[5:8, :, :] = np.tile((1, 1, 1), (3, ret.shape[0], 1))
        return ret

    def get_recomputed_image_matrix_mean(self):
        w = np.tile(np.reshape(self.V, list(self.V.shape) + [1]), [1, 1, 1, 3])
        colors = np.tile(self.b, (self.V.shape[0], self.V.shape[1], 1, 1))
        pixels = np.mean(colors * w, axis=2)
        ret = pixels

        return ret

    def get_recomputed_image_matrix_var(self):
        #c = np.tile(np.reshape((1, 1, 1), (1, 1, -1)), (self.V.shape[0], self.V.shape[1], 1))

        w = self.V
        ret = np.var(w, axis=2)
        ret[:5, :] = np.tile(1, (5, ret.shape[0]))
        #ret[:5, :, :] = np.tile((1, 0, 0), (5, ret.shape[0], 1))

        return ret

    def get_recomputed_image_matrix_valid(self):
        default_color = np.array((0, 1, 0), np.float32)
        #v_plus_default = np.concatenate((np.zeros((self.V.shape[0], self.V.shape[1], 1)), self.V), axis=2)
        v_plus_default = self.V

        V_rounded = np.round(self.V, decimals=0)
        valid_pixels = np.equal(1, np.sum(V_rounded, axis=2)) # true for pixels with one 1 (number > 1)
        valid_pixels = np.tile(np.reshape(valid_pixels, (valid_pixels.shape[0], valid_pixels.shape[1], 1)), (1, 1, 3))
        #print(valid_pixels.shape)


        indexes = np.argmax(v_plus_default, axis=2)
        #ret = np.concatenate((np.reshape(default_color, (1, 3)), self.b))[indexes]
        ret = self.b[indexes]

        mask = np.tile(np.reshape(default_color, (1, 1, 3)), (ret.shape[0], ret.shape[1], 1))

        #print("mask", mask.shape)
        #print("valid_pixels", valid_pixels.shape)

        np.putmask(ret, np.logical_not(valid_pixels), mask)


        #print(f"indexes shape: {indexes.shape}")
        #print(f"recomuputed shape: {ret.shape}")
        #print("indexes", np.mean(indexes, axis=(0, 1)))
        #print("V example:", self.V[10, 10:13, :])

        #print("V mean", np.mean(self.V, axis=(0, 1, 2)))
        #print(self.V)
        return ret




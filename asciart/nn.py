import numpy as np

class NN:
    def __init__(self, layerSizes):
        self.layerSizes = layerSizes

        # initialize weight matrixes
        self.weights = []
        for i in range(len(layerSizes) - 1):
            a = layerSizes[i]
            b = layerSizes[i+1]

            wMatrix = np.random.normal(size=(b, a))
            self.weights.append(wMatrix)

        # init potentials
        self.potentials = [np.zeros((n, 1)) for n in self.layerSizes]
        self.potentialsZ = [np.zeros((n, 1)) for n in self.layerSizes]

    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return NN.sigmoid(z) * (1 - NN.sigmoid(z))

    def propagate(self, input):
        # vector to matrix
        input = np.array([[a] for a in input])
        #input.reshape((len(input), 1))

        self.potentials[0] = input
        self.potentialsZ[0] = np.zeros(input.shape)

        for wi in range(len(self.weights)):
            w = self.weights[wi]
            left = self.potentials[wi]

            z = np.dot(w, left)
            a = NN.sigmoid(z)
            right = NN.sigmoid(z)

            self.potentials[wi+1] = right
            self.potentialsZ[wi+1] = z

        ret = self.potentials[len(self.weights)]
        return [a[0] for a in ret]

    def __str__(self):
        return """=== NN ===
        NN weights: {0}
        NN potentials: {1}
        NN layerSizes: {2}""".format(self.weights, self.potentials, self.layerSizes)



if __name__ == "__main__":
    nn = NN([3, 2, 2])
    input = [1, 2, 3]
    print "Created NN and propagating params {0}...".format(input)
    print "Propagation result:", nn.propagate(input)
    print "NN looks like following now:"
    print nn


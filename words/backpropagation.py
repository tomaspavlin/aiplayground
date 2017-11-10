import numpy as np

from words.nn import NN


def update_net(net, x, y, eta):
    delta_w = backprop(net, x, y)
    net.weights = [w - eta*dw for w, dw in zip(net.weights, delta_w)]

def backprop(net, x, y):
    """Return nabla_w representing the gradient of function C(x).
    This is array of weights deltas"""

    # propagate (feedformward)
    net.propagate(x)

    delta_w = [np.zeros(w.shape) for w in net.weights]
    y = np.array([[a] for a in y])

    # backward
    delta = (net.potentials[-1] - y) * NN.sigmoid_prime(net.potentialsZ[-1])
    delta_w[-1] = np.dot(delta, net.potentials[-2].transpose())

    for l in xrange(2, len(net.weights) + 1):
        delta = np.dot(net.weights[-l + 1].transpose(), delta) * NN.sigmoid_prime(net.potentialsZ[-l])
        delta_w[-l] = np.dot(delta, net.potentials[-l - 1].transpose())

    return delta_w

if __name__ == "__main__":
    net = NN([3, 2, 2])
    x = [1, 2, 3]
    y = [1, 0]
    # y = net.propagate(x)
    # y[0] += 0.01

    print "x={0} y={1}".format(x, y)
    print "backprop:", backprop(net, x, y)
    print "net:", net
    print "performing alg..."

    print "Result is", net.propagate(x)
    for i in range(10000):
        update_net(net, x, y, 0.1)
        if i%1000 == 0:
            print "Result is", net.propagate(x)



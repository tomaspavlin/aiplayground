import test

import backpropagation
import numpy as np
import patterns
import matplotlib.pyplot as plt
import sys

from nn import NN

eta = 0.2
iterations = 20000
test_after = 1000

# works with images 6x14=84
# initialize neural network
net = NN([84, 10, 3, 6])

corr_arr = []

# train nn
for iteration in range(iterations):
    # get training pattern (input and output)
    i, o = patterns.get_pattern()

    # backpropagate the pattern in the network
    backpropagation.update_net(net, i, o, eta)

    if iteration % test_after == 0:
        corr = test.test_network(net)
        print "Iteration:", iteration
        print "Correct test:", corr

        corr_arr.append(corr)


plt.plot(corr_arr)
plt.ylim(0, 1)
plt.show()

# user interaction with network
print "\nNeural network training ended."
print "\nThis is the result:"

while True:
    inputs = patterns.get_outputs_from_image("imgs/test.bmp", 240, 140)

    for inputs_row in inputs:
        for input in inputs_row:
            output = net.propagate(input)
            char = patterns.out2char(output)
            sys.stdout.write(char)

        print ""

    print "Press ENTER for reload"
    sys.stdin.readline()

    print output
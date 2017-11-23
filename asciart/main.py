import test

import backpropagation
import numpy as np
import patterns
import matplotlib.pyplot as plt
import sys

from nn import NN

eta = 0.1
iterations = 50000
test_after = 2000

lambda1 = 1

lambda2After = 40000
lambda2 = 10
eta2 = 0.05


# works with images 6x14=84
# initialize neural network
net = NN([84 + 1, 10, 6])
net.setLambda(lambda1)

#net.setLambda(0.5)

corr_arr = []
lambda_arr = []
eta_arr = []

# train nn
for iteration in range(iterations + 1):
    # get training pattern (input and output)
    i, o = patterns.get_pattern()

    if iteration == lambda2After:
        #net.setLambda(lambda2)
        eta = eta2
        print("Threshold reached")

    if iteration > lambda2After:
        all = iterations-lambda2After
        rel = iteration-lambda2After
        newLambda = lambda1 + 1.0 * (lambda2-lambda1) * rel / all
        net.setLambda(newLambda)


    # backpropagate the pattern in the network
    backpropagation.update_net(net, i, o, eta)



    if iteration % test_after == 0:
        corr = test.test_network(net)
        print "Iteration:", iteration, "/", iterations, "(" + str(100 * iteration / iterations) + "%)"
        print "Correct test:", corr

        corr_arr.append(corr)
        lambda_arr.append(net.getLambda())
        eta_arr.append(eta)



plt.plot(corr_arr, label='corr')
plt.plot([a/10.0 for a in lambda_arr], label='lambda/10')
plt.plot(eta_arr, label='eta')
plt.ylim(0, 1)
plt.legend()
plt.show()

# user interaction with network
print "\nNeural network training ended."
print "\nThis is the result:"

while True:
    inputs = patterns.get_outputs_from_image("imgs/test.bmp", 480, 280)

    for inputs_row in inputs:
        for input in inputs_row:
            output = net.propagate(input)
            char = patterns.out2char(output)
            sys.stdout.write(char)

        print ""

    print "Press ENTER for reload"
    sys.stdin.readline()

    print output
import test

import backpropagation
import numpy as np
import patterns
import matplotlib.pyplot as plt
import sys
import image

from nn import NN
import os

population_size = 5

eta = 0.3
iterations = 20000
test_after = 1000

lambda1 = 1

lambda2After = iterations / 2 #50000
lambda2 = 10#0
eta2 = 0.1


# works with tile sizes 10x10=100
# initialize neural network
hidden = (5, 8)
#hidden = (20)
net = NN([100,
          hidden[0], hidden[1],
          100])
net.setLambda(lambda1)

#net.setLambda(0.5)

corr_arr = []
lambda_arr = []
eta_arr = []

def fac(n):
    ret = 1
    for a in range(n):
        ret *= a+1
    return ret

# print number of posible recognize patterns
n = hidden[1]
print "The initiated neuralnet with {0} neurons in hidden layers can recognize up to {1} patterns (tile types)."\
    .format(hidden, sum([(fac(n)/fac(n-k)/fac(k)) for k in range(1, hidden[0])]))

# train nn
for iteration in range(iterations + 1):

    if iteration == lambda2After:
        #net.setLambda(lambda2)
        eta = eta2
        print("Threshold reached")

    if iteration > lambda2After:
        all = iterations-lambda2After
        rel = iteration-lambda2After
        newLambda = lambda1 + 1.0 * (lambda2-lambda1) * rel / all
        net.setLambda(newLambda)


    # get training pattern (input and output)
    pats = []
    for i in range(population_size):
        pat = patterns.get_random_pattern()
        pats.append((pat, pat))

    #i = o = pat

    # backpropagate the pattern in the network
    #backpropagation.update_net(net, i, o, eta)
    backpropagation.update_net_n(net, pats, eta)



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


# produce images with use of the trained network

while True:

    inputs = image.image2inputs("test.jpg", 480, 280)

    outputs = [net.propagate(input) for input in inputs]
    #print(outputs)
    tile_count = image.get_input_type_count(outputs)

    out_img = image.inputs2image(outputs, 480, 280, True)
    out_img.save("output-round.bmp")

    out_img = image.inputs2image(outputs, 480, 280, False)
    out_img.save("output-unround.bmp")


    print "The generated image consisnts of {0} tile types.".format(tile_count)

    # open out_img

    print "Press ENTER for reload"
    sys.stdin.readline()
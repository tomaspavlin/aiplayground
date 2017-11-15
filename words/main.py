import test

import backpropagation
import numpy as np
import patterns

from nn import NN

eta = 0.2
correctRati = 0.6
iterations = 100000
test_after = 1000

# initialize neural network
net = NN([30, 5, 8])

# train nn
for iteration in range(iterations):
    # get training pattern (input and output)
    if np.random.random() < correctRati:
        i, o = patterns.getRandCorrectPattern()
    else:
        i, o = patterns.getRandIncorrectPattern()

    # backpropagate the pattern in the network
    backpropagation.update_net(net, i, o, eta)

    if iteration % test_after == 0:
        corr = test.test_network_for_correct(net)
        incorr = test.test_network_for_incorrect(net)
        print "Iteration:", iteration
        print "Correct test:", corr
        print "Incorre test:", incorr


# user interaction with network
print "\nNeural network training ended."
print "Now, you can try if the network is working :)"
while True:
    w = raw_input('Write a word (max 6 characters): ')
    if len(w) > 6:
        continue

    input = patterns.encodeWord(w)
    output = net.propagate(input)
    words = patterns.get_words()
    words += ["unknown"]
    print "The network is saying:\n"
    for i in range(len(output)):
        print "\t{0:<8}: {1:.2f}".format(words[i], output[i])

    print "\n>>> I am guessing you typed {0}.".format(words[patterns.interpret_output_i(output)])
    print ""

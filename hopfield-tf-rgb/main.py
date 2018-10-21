import numpy as np

#import visualize
from hopfield import Hopfield
from data import Dataset
from network import Network
import os
import datetime, re
import time

batchsize = int(1e5)

image_filename = "woman.png"
#image_filename = "parrot_sm.png"
#image_filename = "parrot.png"
dataset = Dataset(image_filename)

# colors to use
b = [
    (0, 0, 1),
    (1, 1, 1),
    #(0, 0, 0),
    #(0, 1, 0),
    (1, 0, 0),
    (1, 0.7, 0),
    (72 / 255.0, 57 / 255.0, 37 / 255.0),  # hair color
    (250 / 255.0, 218 / 255.0, 190 / 255.0) # skin color
]

# EP
alpha = 0.4

# EG
beta = 0
# EL is 1 - alpha - beta

strmost = 1
strmost_increase_after = 25000 * 4 * 8
strmost_increase_until = strmost_increase_after * 2
strmost_final = 10

step = 1

# Create logdir name
logdir = "logs/{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

image_matrix = dataset.get_image_matrix_rgb()
network = Network()
network.construct(logdir, ["original", "result", "result_valid", "result_mean", "var"] +
                  [f"result_colors/result_color_{i}" for i in range(len(b))])
network.visualize_image(image_matrix, "original")
hopfield = Hopfield(image_matrix, b=b, alpha=alpha, beta=beta, strmost=strmost)

while True:

    # recompute strmost

    _new_strmost = 1.0 * (step * batchsize - strmost_increase_after) / \
                           (strmost_increase_until - strmost_increase_after) * \
                           (strmost_final - strmost) + strmost

    if _new_strmost > strmost:
        hopfield.strmost = _new_strmost


    print("{}: strmost={:.1f}".format(step * batchsize, hopfield.strmost))

    t = time.time()
    hopfield.recompute_random_batch(batchsize)
    print("Batch took {:.2f}s".format(time.time() - t))


    t = time.time()
    network.visualize_image(hopfield.get_recomputed_image_matrix(), "result")
    network.visualize_image(hopfield.get_recomputed_image_matrix_mean(), "result_mean")
    network.visualize_image(hopfield.get_recomputed_image_matrix_valid(), "result_valid")
    network.visualize_image(hopfield.get_recomputed_image_matrix_var(), "var")
    for i in range(len(b)):
        network.visualize_image(hopfield.get_recomputed_image_matrix_one_color(i), f"result_colors/result_color_{i}")

    print(hopfield.V[10,10])
    print("Visualisation took {:.2f}s".format(time.time() - t))

    step += 1


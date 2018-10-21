import numpy as np

#import visualize
from hopfield import Hopfield
from data import Dataset
from network import Network
import os
import datetime, re


batchsize = 10

#visualize_after = 1000
visualize_image_after = 2000 / batchsize

#matrix = np.random.rand(matrix_h, matrix_w, 3) * (matrix_max_xy - matrix_min_xy) + matrix_min_xy

image_filename = "woman.png"
#image_filename = "parrot.png"
#image_filename = "parrot.png"
dataset = Dataset(image_filename)

#alpha = 0.99
alpha = 0.991

strmost = 1
strmost_increase_after = 25000 * 4
strmost_increase_until = strmost_increase_after * 2
strmost_final = 30




step = 1

# Create logdir name
logdir = "logs/{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

network = Network()
network.construct(logdir, ["original_greyscale", "original_rgb", "result"])

#visualize.copy_image(image_filename, "out_image_original.png")
#visualize.clear_image("out_image.png")

image_matrix = dataset.get_image_matrix_grayscale()
image_matrix_rgb = dataset.get_image_matrix_rgb()


#visualize.visualize_image_matrix_grayscale(image_matrix, "out_image_original_gray.png")
#for i in range(10):
network.visualize_image(image_matrix_rgb, "original_rgb")
network.visualize_image(image_matrix, "original_greyscale")

hopfield = Hopfield(image_matrix, alpha=alpha, strmost=strmost)



while True:

    # recompute strmost

    _new_strmost = 1.0 * (step * batchsize - strmost_increase_after) / \
                           (strmost_increase_until - strmost_increase_after) * \
                           (strmost_final - strmost) + strmost

    if _new_strmost > strmost:
        hopfield.strmost = _new_strmost

    #hopfield.recompute_random()
    hopfield.recompute_random_batch(batchsize)


    if (step) % visualize_image_after == 0:
        print("{}: strmost={:.1f}, image visualization...".format(step, hopfield.strmost))


        new_image_matrix = hopfield.get_recomputed_image_matrix()
        #visualize.visualize_image_matrix_grayscale(new_image_matrix, "out_image.png")
        network.visualize_image(new_image_matrix, "result")
        print("Visualized")


    step += 1
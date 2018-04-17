import numpy as np
import visualize
from hopfield import Hopfield
from data import Dataset



#visualize_after = 1000
visualize_image_after = 5000

#matrix = np.random.rand(matrix_h, matrix_w, 3) * (matrix_max_xy - matrix_min_xy) + matrix_min_xy

#image_filename = "woman.png"
#image_filename = "parrot.png"
image_filename = "parrot.png"
dataset = Dataset(image_filename)

alpha = 0.99

strmost = 1
strmost_increase_after = 25000 * 4
strmost_increase_until = strmost_increase_after * 2
strmost_final = 30



#data = dataset.get_all()
#visualized_data = dataset.get_every_nth(30)

#visualized_data = dataset.next_batch(1000)

step = 0

visualize.copy_image(image_filename, "out_image_original.png")
visualize.clear_image("out_image.png")

image_matrix = dataset.get_image_matrix_grayscale()
hopfield = Hopfield(image_matrix, alpha=alpha, strmost=strmost)
visualize.visualize_image_matrix_grayscale(image_matrix, "out_image_original_gray.png")

# Hopfield.set_strmost() ## TODO



while True:
    step += 1

    #while True:
        #data_sample = np.random.rand(2) * (data_max_xy - data_min_xy) + data_min_xy

        #x, y = data_sample

        #if x < 3 or x > 7 or y<3:
        #    break
        #for i in range(3):
        #    data_sample[0] *= x / matrix_max_xy
        #    data_sample[1] *= y / matrix_max_xy

        #break

    #data_sample = dataset.next_batch(1)[0]

    #if len(data) < data_count:
    #    data.append(data_sample)

    # recompute strmost


    _new_strmost = 1.0 * (step - strmost_increase_after) / \
                           (strmost_increase_until - strmost_increase_after) * \
                           (strmost_final - strmost) + strmost

    if _new_strmost > strmost:
        hopfield.strmost = _new_strmost

    hopfield.recompute_random()



    if step % visualize_image_after == 0:
        print("{}: strmost={:.1f}, image visualization...".format(step, hopfield.strmost))


        new_image_matrix = hopfield.get_recomputed_image_matrix()
        visualize.visualize_image_matrix_grayscale(new_image_matrix, "out_image.png")

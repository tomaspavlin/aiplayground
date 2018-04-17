import numpy as np
import visualize
import kohonen
from data import Dataset


matrix_h = 3
matrix_w = 5

matrix_min_xy = 0
matrix_max_xy = 255

visualize_after = 1000
visualize_image_after = 30000

matrix = np.random.rand(matrix_h, matrix_w, 3) * (matrix_max_xy - matrix_min_xy) + matrix_min_xy

image_filename = "woman.png"
#image_filename = "parrot.png"
dataset = Dataset(image_filename)

data = dataset.get_all()
#visualized_data = dataset.get_every_nth(30)

visualized_data = dataset.next_batch(1000)

step = 0

visualize.copy_image(image_filename, "out_image_original.png")
visualize.clear_image("out_image.png")

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

    data_sample = dataset.next_batch(1)[0]

    #if len(data) < data_count:
    #    data.append(data_sample)

    kohonen.alg_step(matrix, data_sample, step)

    if step % visualize_after == 0:
        text = "Step={}\nalpha={:.2f}\nsigma={:.2f}".format(step, kohonen.alpha(step), kohonen.sigma(step))
        print(text.replace("\n", ", "))
        visualize.visualize_grid(matrix, visualized_data, "out_grid.png", text)
        visualize.visualize_palette(matrix, "out_palette.png")

    if step % visualize_image_after == 0:
        print("Image visualization")
        visualize.visualize_image(data, dataset.image_size, "out_image.png", matrix)


    # TODO: image compressor, better visualization of 3D, then maybe visualise the map also with colors
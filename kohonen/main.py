import numpy as np
import visualize
import kohonen
import threading


matrix_h = 64
matrix_w = 1
data_count = 2000

matrix_min_xy = 0
matrix_max_xy = 10

data_min_xy = 0
data_max_xy = 10

visualize_after = 200

matrix = np.random.rand(matrix_h, matrix_w, 2) * (matrix_max_xy - matrix_min_xy) + matrix_min_xy
#data = np.random.rand(data_count, 2) * (data_max_xy - data_min_xy) + data_min_xy
data = []

step = 0

while True:
    step += 1

    while True:
        data_sample = np.random.rand(2) * (data_max_xy - data_min_xy) + data_min_xy
        x, y = data_sample

        #if x < 3 or x > 7 or y<3:
        #    break
        #for i in range(3):
        #    data_sample[0] *= x / matrix_max_xy
        #    data_sample[1] *= y / matrix_max_xy

        break

    if len(data) < data_count:
        data.append(data_sample)

    kohonen.alg_step(matrix, data_sample, step)

    if step % visualize_after == 0:
        text = "Step={}\nalpha={:.2f}\nsigma={:.2f}".format(step, kohonen.alpha(step), kohonen.sigma(step))
        print(text.replace("\n",", "))
        visualize.visualize(matrix, data, text)

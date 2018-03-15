import numpy as np

alpha_p = 0.2
sigma_p = 0.4
sigma_coef = 30

def prog(step, p):
    #p = 0.4
    ret = np.power(step, -p)
    return ret

def alpha(step):
    return prog(step, alpha_p)

def sigma(step):
    return prog(step, sigma_p) * sigma_coef

def neighbourhood_function(u, v, step):
    sig = sigma(step)

    dist = np.linalg.norm(u - v)

    ret = np.exp(-dist/sig/sig)
    return ret

def alg_step(matrix, data_sample, step):
    # get random sample
    #rand_item = data[np.random.randint(data_count)]
    rand_item = data_sample

    nearest_x = 0
    nearest_y = 0
    nearest_dist = np.inf

    # find nearest neuron
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            neur = matrix[y][x]
            dist = np.linalg.norm(neur - rand_item)

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_x = x
                nearest_y = y


    # modify neurons
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            matrix[y][x] += alpha(step) * \
                            neighbourhood_function(np.array([x, y]), np.array([nearest_x, nearest_y]), step) * \
                            (rand_item - matrix[y][x])


if __name__ == "__main__":
    m = np.random.random_integers(0, 10, (10, 10, 2)) / 10.0
    #m = np.array([[[1,2],[2,3]]])
    d = np.array([2.0, 2.0], dtype=np.float32)
    s = 1
    while True:
        for i in range(100):
            alg_step(m, d, s)
            s += 1
        print(s)

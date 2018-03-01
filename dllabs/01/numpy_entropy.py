#!/usr/bin/env python3
import numpy as np

def H(P, Q = None):
    if Q == None:
        # P = filter(lambda a : (a != 0), P)
        Q = P

    # hack
    logq = np.array([-np.inf if q == 0 else np.log(q) for q in Q])
    #print(logq)

    #ret = - np.sum(filter(lambda a: not np.isnan(a), np.multiply(P,logq)))
    ret = - np.sum(P*logq)
    #print(ret)
    return ret

def KLD(P, Q):
    return H(P, Q) - H(P)

if __name__ == "__main__":

    counts = {}
    countsum = 0

    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")

            if line in counts:
                counts[line] += 1
            else:
                counts[line] = 1

            countsum += 1

    data_dist_map = {}
    for w in counts:
        data_dist_map[w] = float(counts[w])/countsum

    data_dist = np.array([data_dist_map[a] for a in data_dist_map], dtype=np.float32)

    # print(data_dist)


    model_dist_map = {}

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line

            arr = line.split("\t")
            if len(arr) != 2:
                print("Input error")
                exit(1)

            w = arr[0]
            p = float(arr[1])

            model_dist_map[w] = p

    entropy = H(data_dist)
    print("{:.2f}".format(entropy))


    #data_dist = [data_dist_map[w] if w in data_dist_map else 0 for w in model_dist_map]
    #model_dist = [model_dist_map[w] for w in model_dist_map]
    model_dist = [model_dist_map[w] if w in model_dist_map else 0 for w in data_dist_map]

    print("{:.2f}".format(H(data_dist, model_dist)))

    print("{:.2f}".format(KLD(data_dist, model_dist)))

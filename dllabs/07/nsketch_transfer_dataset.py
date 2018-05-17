#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        data = data['arr_0'].item()


        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._features = data["features"] if "features" in data else None

        #print(filename)
        #print(self._images.shape)

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))


    def save(self, filename):
        data = dict()
        data["images"] = self._images
        if self._labels is not None:
            data["labels"] = self._labels

        if self._features is not None:
            data["features"] = self._features

        np.savez_compressed(filename, data)

    def visualize(self):
        import matplotlib.pyplot as plt
        for image in self._images:
            image = image.reshape(image.shape[0], image.shape[1])
            plt.gray()
            plt.imshow(image)
            plt.show()
            print(image.shape)

    @property
    def images(self):
        return self._images

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], \
               self._features[batch_perm] if self._features is not None else None, \
               self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))
            return True
        return False

    def set_features(self, features):
        self._features = features


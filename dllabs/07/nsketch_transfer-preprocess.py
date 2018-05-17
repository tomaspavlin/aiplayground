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


class Network:
    WIDTH, HEIGHT = 224, 224
    LABELS = 250
    FEATURES = 1056

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None, is_training=False)
            self.nasnet_saver = tf.train.Saver()



            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            #self.features = tf.identity(features)
            self.features = tf.placeholder(dtype=tf.float32, shape=(None, self.FEATURES))

            #self.features = tf.layers.dropout(self.features, training=self.is_training, name="dropout")

            #hidden_layer = tf.layers.dense(self.features, 1024, activation=tf.nn.relu, name="hidden_layer")
            hidden_layer = self.features

            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name="output_layer")


            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step)


            self.predictions = tf.argmax(output_layer, axis=1, name="predictions")


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

    def train_batch(self, images, features, labels):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images,
                          self.features: features,
                          self.labels: labels,
                          self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_features, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy], {
                    self.images: batch_images,
                      self.features: batch_features,
                      self.labels: batch_labels,
                      self.is_training: False})
            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
            #print(batch_accuracy, batch_accuracy * len(batch_images) / len(dataset.images), accuracy)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})
        return accuracy

    def predict(self, dataset, batch_size):
        labels = []
        i = 0
        while not dataset.epoch_finished():
            i += batch_size
            print(i)
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)

    def predict_features(self, dataset, batch_size):
        features = []
        i = 0
        while not dataset.epoch_finished():
            i += batch_size
            print(i)
            images, _ = dataset.next_batch(batch_size)
            features.append(self.session.run(self.features, {self.images: images, self.is_training: False}))
        return np.concatenate(features)

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--nasnet", default="nets/nasnet/model.ckpt", type=str, help="NASNet checkpoint path.")
    parser.add_argument("--threads", default=1, type=int, help=" Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    #train = Dataset("nsketch-train.npz")
    #dev = Dataset("nsketch-dev.npz", shuffle_batches=False)
    #test = Dataset("nsketch-test.npz", shuffle_batches=False)

    train = Dataset("nsketch-train-features.npz", shuffle_batches=False)
    dev = Dataset("nsketch-dev-features.npz", shuffle_batches=False)
    test = Dataset("nsketch-test-features.npz", shuffle_batches=False)

    #dev.visualize()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    print("constructed")

    # uncomment for preprocessing features
    # preprocessing features
    #a = [(dev, "dev"), (test, "test"), (train, "train")]
    a = [(train, "train")]
    for dataset, name in a:
        print("Predicting features for {} ({})".format(name, dataset.images.shape[0]))
        features = network.predict_features(dataset, args.batch_size)
        dataset.set_features(features)
        dataset.save("nsketch-{}-features.npz".format(name))


    print("Training...")

    # Train
    for i in range(args.epochs):
        print("Epoch {}".format(i + 1))
        while not train.epoch_finished():
            images, features, labels = train.next_batch(args.batch_size)
            #print(".")
            network.train_batch(images, features, labels)
            #print("..")

        accur = network.evaluate("dev", dev, args.batch_size)
        print("{:.1f}".format(accur * 100))

    # Predict test data
    with open("{}/nsketch_transfer_test.txt".format(args.logdir), "w") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)

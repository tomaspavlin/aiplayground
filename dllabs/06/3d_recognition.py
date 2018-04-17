#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._voxels = data["voxels"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False


class Network:
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name="voxels")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")


            #filters = [2, 4, 8, 16, 32]
            filters = [8**i for i in range(1,10)]
            print(filters)

            conv_layer = self.voxels
            for i in range(3):
                for j in range(1):
                    conv_layer = tf.layers.conv3d(
                        conv_layer,
                        filters=filters[i],
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        activation=None,
                        name="convolution_{}_{}".format(i, j)
                    )

                    #conv_layer = tf.layers.batch_normalization(
                    #    conv_layer,
                    #    training=self.is_training
                    #)

                    conv_layer = tf.nn.relu(conv_layer)

                #conv_layer = tf.layers.max_pooling3d(
                #    conv_layer,
                #    pool_size=3,
                #    strides=2,
                #    padding='same',
                #    name="maxpooling_{}".format(i)
                #)

            hidden_layer = tf.layers.flatten(conv_layer, name="flatten_layer")

            hidden_layer = tf.layers.dense(hidden_layer, 32, activation=tf.nn.relu, name="dense_layer")

            hidden_layer = tf.layers.dropout(hidden_layer, rate=0.5, name="dropout_layer")

            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name="output_layer")

            #output_layer = tf.zeros_like(output_layer)

            #print(output_layer)
            #print(self.labels)

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)


            self.predictions = tf.argmax(output_layer, axis=1, name="predictions")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            # Summaries
            #confusion_matrix = tf.confusion_matrix(self.labels, self.predictions, dtype=tf.float32)
            #print(confusion_matrix)

            #confusion_matrix = tf.reshape(
            #    confusion_matrix,
            #    shape=[-1, self.LABELS, self.LABELS, 1])


            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
                                           #tf.contrib.summary.image("train/confusion_matrix", confusion_matrix)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

                                               #tf.contrib.summary.image("train/confusion_matrix", confusion_matrix)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries["train"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, voxels, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.voxels: voxels, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--modelnet_dim", default=32, type=int, help="Dimension of ModelNet data.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_split", default=0.9, type=float, help="Ratio of examples to use as train.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset("modelnet{}-train.npz".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset("modelnet{}-test.npz".format(args.modelnet_dim), shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train

    j = 0
    for i in range(args.epochs):
        print("EPOCH", i + 1)
        while not train.epoch_finished():
            print(j + 1)

            voxels, labels = train.next_batch(args.batch_size)
            network.train(voxels, labels)

            j += 1

            if (j + 1) % 20 == 0:
                print(dev.voxels.shape)
                accuracy = network.evaluate("dev", dev.voxels, dev.labels)
                print("{:.2f}".format(accuracy * 100))



            if (j + 1) % 100 == 0:
                print("Predicting test data")
                accuracy = network.evaluate("dev", dev.voxels, dev.labels)
                # Predict test data
                with open("{}/3d_recognition_test_{}_acc{:.2f}.txt".format(args.logdir, i + 1, accuracy*100), "w") as test_file:
                    while not test.epoch_finished():
                        voxels, _ = test.next_batch(args.batch_size)
                        labels = network.predict(voxels)

                        for label in labels:
                            print(label, file=test_file)
                print("Predicting test data has ended")

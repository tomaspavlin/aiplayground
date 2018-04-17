#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
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
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            self.dropout_rate = tf.placeholder(tf.float32, [], name="dropout_rate")

            # Computation
            flattened_images = tf.layers.flatten(self.images, name="flatten")


            hidden_layer = tf.layers.dense(flattened_images, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer")

            # TODO: Implement dropout on the hidden layer using tf.layers.dropout,
            # with using dropout date of args.dropout. The dropout must be active only
            # during training -- use `self.is_training` placeholder to control the
            # `training` argument of tf.layers.dropout. Store the result to `hidden_layer_dropout`.

            #bb = tf.multiply(hidden_layer, (1 - self.dropout_rate), name="mul")

            #hidden_layer_dropout = a #tf.cond(self.is_training, lambda: a, lambda: bb)

            hidden_layer_dropout = tf.layers.dropout(hidden_layer, self.dropout_rate, training=self.is_training, name="hidden_layer_dropout")

            self.output_layer = tf.layers.dense(hidden_layer_dropout, self.LABELS, activation=None, name="output_layer")
            #self.mean = tf.
            # not working, maybe bad multiplication coef?
            self.predictions = tf.argmax(self.output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, rate):

        ret = self.session.run([self.output_layer, self.training, self.summaries["train"]], {self.images: images, self.labels: labels,
                                                                    self.dropout_rate: rate,
                                                                    self.is_training: True})[0]

        #print(ret)
        return ret

    def evaluate(self, dataset, images, labels, rate):
        return self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels,
                                                   self.is_training: False,
                                                   self.dropout_rate: rate})[0]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--training_size", default=5000, type=int, help="Number of training images.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    # Note that because of current ReCodEx limitations, the MNIST dataset must
    # be read from the directory of the script -- that is why the "mnist_data/"
    # from `mnist_example.py` has been changed to current directory ".".
    #
    # Additionally, loading of the dataset prints to stdout -- this loading message
    # is part of expected output when evaluating on ReCodEx.
    mnist = mnist.input_data.read_data_sets(".", validation_size=60000-args.training_size, reshape=False, seed=42)
    batches_per_epoch = mnist.train.num_examples // args.batch_size

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        for b in range(batches_per_epoch):
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels, args.dropout)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels, args.dropout)

    accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels, args.dropout)
    print("{:.2f}".format(100 * accuracy))

    accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels, 0)
    print("{:.2f}".format(100 * accuracy))

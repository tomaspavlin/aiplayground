#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    OBSERVATIONS = 4
    ACTIONS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")


            # mine
            hidden_layer_size = 10
            hidden_layer_count = 1

            hidden_layer = self.observations

            for i in range(hidden_layer_count):
                hidden_layer = tf.layers.dense(hidden_layer, hidden_layer_size, activation=tf.nn.relu, name="hidden_layer_" + str(i))

            output_layer = tf.layers.dense(hidden_layer, self.ACTIONS, activation=None, name="output_layer")

            self.actions = tf.argmax(output_layer, axis=1, name="actions")

            # Global step
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.actions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = [tf.contrib.summary.scalar("train/loss", loss),
                                  tf.contrib.summary.scalar("train/accuracy", accuracy)]
                self.test_summaries = tf.contrib.summary.scalar("test/loss", loss)

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/actions", self.actions)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, observations, labels):
        self.session.run([self.training, self.summaries], {self.observations: observations, self.labels: labels})

    def save(self, path):
        self.saver.save(self.session, path)

    def test(self, observations, labels):
        self.session.run([self.test_summaries], {self.observations: observations, self.labels: labels})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[0:4]])
            labels.append(int(columns[4]))
    observations, labels = np.array(observations), np.array(labels)

    # TODO: splite to testing and training data
    testsize = 0 # out of 100 data

    observations_test, labels_test = observations[:20], labels[:20]
    observations, labels = observations[20:], labels[20:]

    print("datalen: {}, {}".format(labels.shape, labels_test.shape))

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    batch_size = 1
    batches_per_epoch = len(observations) // batch_size

    for i in range(args.epochs):
        for ii in range(batches_per_epoch):
            # get random batch
            batch = np.random.choice(len(observations), batch_size)
            observations_b, labels_b = observations[batch], labels[batch]

            # train step
            network.train(observations_b, labels_b)

        #network.test(observations_test, labels_test)
        network.test(observations, labels)


    # Save the network
    network.save("gym_cartpole/model")

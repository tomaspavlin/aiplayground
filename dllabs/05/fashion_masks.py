#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None, self._masks[batch_perm] if self._masks is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))
            return True
        return False

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
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            #fil = args.filters

            # convolutions
            conv_layer = self.images
            for i in range(args.convolutions):
                conv_layer = tf.layers.conv2d(
                    conv_layer,
                    filters=args.conv_filters,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name="conv_layer_{}".format(i + 1),
                    activation=tf.nn.relu
                )

                #continue
                conv_layer = tf.layers.max_pooling2d(
                    conv_layer,
                    pool_size=3,
                    strides=2,
                    padding='same',
                    name="max_pooling_{}".format(i + 1)
                )

                conv_layer = tf.layers.batch_normalization(
                    conv_layer,
                    training=self.is_training,
                    name="batch_norm_{}".format(i + 1)
                )

            flatten_layer = tf.layers.flatten(conv_layer, name="flatten_layer")

            # dense layers
            dense_layer = flatten_layer
            for i in range(1):
                dense_layer = tf.layers.dense(
                    dense_layer,
                    units=100,
                    activation=tf.nn.relu,
                    name="dense_layer_{}".format(i)
                )

            dense_layer = tf.layers.dropout(
                dense_layer,
                rate=0.5,
                training=self.is_training,
                name="dense_layer_dropout"
            )

            output_layer_labels = tf.layers.dense(
                dense_layer,
                activation=None,
                # units=self.WIDTH*self.HEIGHT + self.LABELS
                units=self.LABELS
            )

            #output_layer_masks = tf.layers.dense(
            #    dense_layer,
            #    activation=None,
            #    units=self.WIDTH*self.HEIGHT
            #)

            output_layer_masks = tf.layers.conv2d(
                    conv_layer,
                    filters=1,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    name="conv_layer_output_layer_masks",
                    activation=tf.nn.relu
                )

            # conv transpose
            conv_transpose_layer = conv_layer
            for i in range(args.convolutions):
                conv_transpose_layer = tf.layers.conv2d_transpose(
                        conv_transpose_layer,
                        filters=10,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        name="conv_layer_trans_{}".format(i + 1),
                        activation=tf.nn.relu
                    )

            output_layer_masks = tf.layers.conv2d(
                        conv_transpose_layer,
                        filters=1,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        name="output_layer_masks_conv_1x1",
                        activation=tf.nn.relu
                    )

            #output_layer_masks = tf.reshape(output_layer_masks, shape=[-1, self.HEIGHT, self.WIDTH, 1], name="masks_reshaped")


            loss_labels = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer_labels, scope="loss_labels")
            loss_masks = tf.losses.mean_squared_error(self.masks, output_layer_masks, scope="loss_masks")

            loss = loss_labels + loss_masks
            #loss = loss_masks
            #loss = loss_labels

            global_step = tf.train.create_global_step()

            batchnorm_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(batchnorm_update):
                self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")


            self.labels_predictions = tf.argmax(output_layer_labels, axis=1)
            self.masks_predictions = tf.round(output_layer_masks, name="masks_predictions_rounded")



            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            self.iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/loss_labels", loss_labels),
                                           tf.contrib.summary.scalar("train/loss_masks", loss_masks),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                                           tf.contrib.summary.scalar("train/iou", self.iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/loss_labels", loss_labels),
                                               tf.contrib.summary.scalar(dataset+"/loss_masks", loss_masks),
                                               tf.contrib.summary.scalar(dataset+"/accuracy", self.accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", self.iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        ret, _ = self.session.run([self.iou, self.summaries[dataset]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})
        return ret

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=40, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--convolutions", default=2, type=int, help="Conv layers count.")
    parser.add_argument("--conv_filters", default=50, type=int, help="Number of filters in each CL.")
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
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz", shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        print("Epoch {}".format(i+1))
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(args.batch_size)
            network.train(images, labels, masks)

        iou = network.evaluate("dev", dev.images, dev.labels, dev.masks)
        print("{:.2f}".format(iou*100))

        # Predict test data
        with open("{}/fashion_masks_test_{}.txt".format(args.logdir, i + 1), "w") as test_file:
            while not test.epoch_finished():
                images, _, _ = test.next_batch(args.batch_size)
                labels, masks = network.predict(images)
                for i in range(len(labels)):
                    print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)



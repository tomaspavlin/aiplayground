#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index

            # mine
            # added digits
            for a in range(10):
                alphabet_map[str(a)] = 2
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet: break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.bool)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map: char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = self._text[i].isupper()

        # Compute alphabet
        self._alphabet = [""] * (len(alphabet_map) - 9) #minus digits
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        #print(alphabet_map)
        #print(self._alphabet)

        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.int32)
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i]
        return batch_windows, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False


class Network:
    HIDDEN_LAYERS_A = 1
    HIDDEN_LAYERS_B = 0
    HIDDEN_LAYER_SIZE = 50
    DROPOUT_RATE = 0

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.windows = tf.placeholder(tf.int32, [None, 2 * args.window + 1], name="windows")
            self.labels = tf.placeholder(tf.int32, [None], name="labels") # Or you can use tf.int32, bool
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Define a suitable network with appropriate loss function
            self.input_layer = tf.one_hot(self.windows, args.alphabet_size, name="input_layer", dtype=tf.float32)
            flattened_input = tf.layers.flatten(self.input_layer)

            hidden = flattened_input #tf.cast(self.windows, tf.float32)

            for i in range(self.HIDDEN_LAYERS_A):
                hidden = tf.layers.dense(hidden, self.HIDDEN_LAYER_SIZE, activation=tf.nn.relu, name=f"hidden_A{i}")

            # maybe add inbetween the hidden layers
            if self.DROPOUT_RATE > 0:
                hidden = tf.layers.dropout(hidden, rate=self.DROPOUT_RATE, training=self.is_training, name="dropout")


            for i in range(self.HIDDEN_LAYERS_B):
                hidden = tf.layers.dense(hidden, self.HIDDEN_LAYER_SIZE, activation=tf.nn.relu, name=f"hidden_B{i}")

            output = tf.layers.dense(hidden, 2, activation=None, name="output")  # just for one sample

            self.predictions = tf.argmax(output, axis=1, output_type=tf.int32)  # why 1?

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output, scope="my_loss")
            #loss = tf.losses.compute_weighted_loss()


            # TODO: Define training

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step = global_step, name="training")


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

    def train(self, windows, labels):
        self.session.run([self.training, self.summaries["train"]], {self.windows: windows, self.labels: labels,
                                                                    self.is_training: True})

    def evaluate(self, dataset, windows, labels):
        return self.session.run([self.predictions, self.accuracy, self.input_layer, self.summaries[dataset]], {
            self.windows: windows, self.labels: labels, self.is_training: False})

def generate(network, dataset, outputfile):
    print("Generating uppercased data")
    test_windows, xx = dataset.all_data()
    test_labels = network.evaluate("test", test_windows, xx)[0]

    with open(outputfile, "w") as f:
        for c, l in zip(dataset.text, test_labels):
            isup = l == 1
            # c = test.alphabet[ci]
            if isup:
                c = c.upper()
            else:
                c = c.lower()
            # print(c + " " + str(l))
            f.write(c)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("--alphabet_size", default=70, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=3000, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=3, type=int, help="Size of the window to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself


    #alphabet = 70
    #alphabet_size = alphabet

    alphabet = ['<pad>', '<unk>', ' ', '.', ',', '\n', '-', ':', '"', '/', '!', '|', '?']
    alphabet += ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    alphabet_size = len(alphabet)


    args.alphabet_size = alphabet_size
    #alphabet_size = alphabet

    # Load the data
    train = Dataset("uppercase_data_train.txt", args.window, alphabet=alphabet)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=alphabet)

    print(f"Train alphabet: '{train.alphabet}'")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    dev_windows, dev_labels = dev.all_data()

    np.set_printoptions(threshold=np.nan)

    ii = 0
    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            windows, labels = train.next_batch(args.batch_size)
            network.train(windows, labels)

            if ii% 300 == 0:
                arr = network.evaluate("dev", dev_windows, dev_labels)
                accuracy = arr[1]
                print(f"Epoch {i+1}/{args.epochs}: {accuracy*100:.2f}")

                #print(f"window: {dev_windows[0]}")
                #print(f"input: {arr[2][0]}")

                # dev_windows, dev_labels = dev.all_data()
                # accuracy = network.evaluate("dev", dev_windows, dev_labels)[1]

                # print(f"Epoch {i+1}/{args.epochs}: {accuracy*100:.2f}")
            if ii % 1200 == 0:
                # TODO: Generate the uppercased test set
                generate(network, test, "uppercase_data_test_output.txt")
                generate(network, dev, "uppercase_data_dev_output.txt")
                print("===")

            ii += 1




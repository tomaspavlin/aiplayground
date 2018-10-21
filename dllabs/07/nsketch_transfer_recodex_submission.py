# coding=utf-8

source_1 = """#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet

from nsketch_transfer_dataset import Dataset


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
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

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

            hidden_layer = tf.layers.dropout(self.features, training=self.is_training, name=\"dropout\")

            #hidden_layer = tf.layers.dense(self.features, 1024, activation=tf.nn.relu, name=\"hidden_layer\")
            #hidden_layer = self.features

            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name=\"output_layer\")


            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")

            self.my_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=100)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step)


            self.predictions = tf.argmax(output_layer, axis=1, name=\"predictions\")


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name=\"given_loss\")
                self.given_accuracy = tf.placeholder(tf.float32, [], name=\"given_accuracy\")
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

            # Load Mine
            #if args.model:
            #    self.my_saver.restore(self.session, args.model)

    def train_batch(self, images, features, labels):
        self.session.run([self.training, self.summaries[\"train\"]],
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
            #print(i)
            _, features, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.features: features, self.is_training: False}))
        return np.concatenate(labels)

    def save_mine(self, filename, step):
        self.nasnet_saver.save(self.session, filename, global_step=step, write_meta_graph=False)

if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=None, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=None, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--nasnet\", default=\"nets/nasnet/model.ckpt\", type=str, help=\"NASNet checkpoint path.\")
    parser.add_argument(\"--model\", default=None, type=str, help=\"NASNet checkpoint path.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\" Maximum number of threads to use.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value)
                  for key, value in sorted(vars(args).items()))).replace(\"/\", \"-\")
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    #train = Dataset(\"nsketch-train.npz\")
    #dev = Dataset(\"nsketch-dev.npz\", shuffle_batches=False)
    #test = Dataset(\"nsketch-test.npz\", shuffle_batches=False)

    train = Dataset(\"nsketch-train-features.npz\")
    dev = Dataset(\"nsketch-dev-features.npz\", shuffle_batches=False)
    test = Dataset(\"nsketch-test-features.npz\", shuffle_batches=False)

    #dev.visualize()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    print(\"constructed\")


    print(\"Training...\")

    # Train
    for i in range(args.epochs):
        print(\"Epoch {}\".format(i + 1))
        while not train.epoch_finished():
            images, features, labels = train.next_batch(args.batch_size)
            #print(\".\")
            network.train_batch(images, features, labels)
            #print(\"..\")

        accur = network.evaluate(\"dev\", dev, args.batch_size)
        print(\"{:.1f}\".format(accur * 100))

        #network.save_mine(\"nets/mine/model_b\", i + 1)

        # Predict test data
        with open(\"{}/nsketch_transfer_test_{}_{:.1f}.txt\".format(args.logdir, i + 1, accur * 100), \"w\") as test_file:
            labels = network.predict(test, args.batch_size)
            for label in labels:
                print(label, file=test_file)
"""

source_2 = """#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet

from nsketch_transfer_dataset import Dataset

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
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

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

            self.features = tf.identity(features)

            #self.features = tf.layers.dropout(self.features, training=self.is_training, name=\"dropout\")

            #hidden_layer = tf.layers.dense(self.features, 1024, activation=tf.nn.relu, name=\"hidden_layer\")
            hidden_layer = self.features

            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name=\"output_layer\")


            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step)


            self.predictions = tf.argmax(output_layer, axis=1, name=\"predictions\")


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name=\"given_loss\")
                self.given_accuracy = tf.placeholder(tf.float32, [], name=\"given_accuracy\")
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

    def train_batch(self, images, features, labels):
        self.session.run([self.training, self.summaries[\"train\"]],
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
            images, _, _ = dataset.next_batch(batch_size)
            features.append(self.session.run(self.features, {self.images: images, self.is_training: False}))
        return np.concatenate(features)

if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=None, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=None, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--nasnet\", default=\"nets/nasnet/model.ckpt\", type=str, help=\"NASNet checkpoint path.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\" Maximum number of threads to use.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value)
                  for key, value in sorted(vars(args).items()))).replace(\"/\", \"-\")
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    #train = Dataset(\"nsketch-train.npz\", shuffle_batches=False)
    #dev = Dataset(\"nsketch-dev.npz\", shuffle_batches=False)
    #test = Dataset(\"nsketch-test.npz\", shuffle_batches=False)

    train = Dataset(\"nsketch-train-features.npz\", shuffle_batches=False)
    dev = Dataset(\"nsketch-dev-features.npz\", shuffle_batches=False)
    test = Dataset(\"nsketch-test-features.npz\", shuffle_batches=False)

    #dev.visualize()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    print(\"constructed\")

    # uncomment for preprocessing features
    # preprocessing features
    #a = [(dev, \"dev\"), (test, \"test\"), (train, \"train\")]
    a = [(train, \"train\")]
    for dataset, name in a:
        print(\"Predicting features for {} ({})\".format(name, dataset.images.shape[0]))
        features = network.predict_features(dataset, args.batch_size)
        dataset.set_features(features)
        dataset.save(\"nsketch-{}-features.npz\".format(name))

"""

source_3 = """#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        data = data['arr_0'].item()


        self._images = data[\"images\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None
        self._features = data[\"features\"] if \"features\" in data else None

        #print(filename)
        #print(self._images.shape)

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))


    def save(self, filename):
        data = dict()
        data[\"images\"] = self._images
        if self._labels is not None:
            data[\"labels\"] = self._labels

        if self._features is not None:
            data[\"features\"] = self._features

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
        return self._images[batch_perm], \\
               self._features[batch_perm] if self._features is not None else None, \\
               self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(len(self._images))
            return True
        return False

    def set_features(self, features):
        self._features = features

"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;2XsY0$l(YjsrCpM-yTVCoksXiB=c(R2<khzLZHAOd$z`fit1&3ujwOCMUm3KKEu(blIdNGJN~uW<Ko6s9|_)gaWkRe{hukFJ8~#iQNL>EwViS)S4%@H-NzQ_=3@hQ>bxr_F2R{G=Qz1--%bVYhRs{>{;-Vl`?rAoD?lyn7KG?!T!eZ&+0x1JwDOP&fGy<caC_(U;|R}QlaV$0KLhYrY*X^$nK2^eUhA3qSdl_UtIO;m^M=wck<BkG;+%7P_vwqJTP*DnZM}Rh)Anq|3C<WbIN4G+wq+s>Jo12O6!XNKjWfA7_~eWqpAd9^ztpovRS<5Hzh7mPNOdnR}mGC&%wl19P`@c*UrQ(Hku%WIfmc2E*y%j5$mr<tZtF8(f%N52$H80VrTJQel{an*69zDK}nmbD2@guG_Mm2%I2{iahml@$)=8RmoXt*2(KRW-X+MSN8IgS)gsF5`P!vkTauJ!MO+s<IlL?<<S9BG6VOMil=`9QRR2VTTDFwqEUfE8OzPFI1x6}mBr}B;(@2ZBJ6t4>XX~>^Gq}Y#-v4K^mEephisjp*rq~wo`xEzIt~q49)dPjyQ$_XXI|yhRxFM#+E3f1L{!vuA&-5LRLApOQ3tQ^7f>zbLh2H#MntGIrQ1eQqL-LQdaOX`fWUs8}qBgUFTWPp^;%|$&%2y4Dv154<W)4_%W{dO4SWV})CMaQ}7Ax3y=8I^X6E`M+T;;Uw&{3qeNin?G(J;fENh%=H#Zi>jQ#T>%6{E;y>OVy8F!LSU7LIckN!m?hfOsR2QEiY)(KEs9WO-kS&6KIKI?yo+N(e5pxQ>EQF^Yq9$v2G`s1zZ{{{zo#N{A36er%1PD=bQ{C!>F3<VFHDWz=<9#Ag`>P@bGv45#KouvQSRx}rSBWjGL(6*!8z@<gskV}to7TLM@$8^5=NvA|i74~Y%;fPIO!4tzItp0mzwXnuC8(JH`^l53;UszI?_&nk>|Q?%ML<6)sAW&ONuq1Zl}1$@&+LYOFTd2*R+j{BGnHK^erM}oi|ZVhHIETjeSULUSd4>}zCGm>^p`L5(F>U216e+9I`LZR1!2=^Wlr=fi59Uj<W1nx_FNbmR!1rSvKl<o3r#w-#S^DKeB6#+^r@ff-hEAun`9sqBa(rJD*BhTQsqmeeR5C#$?h-WiC*k|P`pFH$-3?N8irXQ|Bh5A97rg$l0A3Js$T9WYjU%%%tRFO<gXlF4uV>7LU)k-JFJY>2|tO$=IvRhh)M?)(4<)Iv$;w!xUSezF5$D*ma0@N5C(#&<mat^rVYnO+GoJUL<?vS6y?q-7WZ#&g;M{80u$A+L7V~^?`AB|@39>Gg`q=A$gN&GWX(U<!b?x=s-H@ys!mD5r!+0i=AX&+_SAbmX?d5`}HbF%$r3!A>>5MHywX*U!;a<KS|aih(~Z}RE~fR-QEA%R6XACj5zMI1QKVG>qw12t4xR`kc(FyQTJ>aevtqd1yl0G>M%{T&nN{DW*O#_X?m5e)kbW}}azH>L&ul|_ET`R;uKebV~UI&8z|vbKN5$P;?~K5pfj=^!u+8KzC#bccQ8BRVYx!uKk9Ma7}9GzardU;A<%4J;-)&k^1e_LaTPih?Y2+;Qi*6w(8=ncck;Gf=U*7;3?-7Kk%vkoWMsL6^;bp4;<t?d-Z>QrE8vL72~V+lriNiw>?oEx;R;V(u1s1VyCHcU?V;Y329EyFJYAgrNoXuS%FNoQ}KYS6f;Plj{>F@}hUTqz~nY9Mok2*8-h{M%RP#Y;=%#!@2`HsSUj-Cn4!Ax<9q3?okOti$w+4*@`!v+~{?5Zoa!=gV~vv4xrAERu@G2m9f26rtnF)gLN3-r+jUcm`k!eCHujz$q+>|I%s!#%?B)C!+aFmjAqc{ZaeYRp>BR~n}gf#-)1rc5wKeR%Nqwy!h}}$)LOM@J>iYh9syiWEo7)>TBT<uHy~3JGo#l{h1)>N2LMCBk4%irOo!VJf7TeFwNwi#zLtTi`=fRxHx%OqHF)*D9ewSAO&%5P@faoE$DwCRvu1TR90wVg!RZ3z&^{4k%E@Qb-DGcDG1PWsW6^mP3FcFW3FV}^>K97=0T(N8EU|YEbZ8F1`uXGpHSdBHR$f)xT-KvqI7A`fbn+axk)zBqJ@Ldnm$X|36*rrKxOCyXIF#>hx1hbP;_P)FBkHxYaW0TKwg(v?+&jVof~7{jHH6$LFmK7!s_uD0>@Ic*yFMT$m-S7#6~wPusbvO!*(TS_J!IjGiqY*fJ)VtcPLioKXKXsX0N7z#cFsv6_PfTdj^Piyaa|{kereLt&-}M<U+zFCR?x;Ktpnb(mFkJJ%#G>U#VJEP9x9RD(bcaz#-S2P8c3Pp>eZLL7IgI)&re)R;cY?^#V(5c7`%DX*IE@kW`hh$UH$*3Z2QsU(F=r@py#i{9G&46yCl^JsR-w1AW;tOO(>0wJyJ7j$IB=^5TNNNRnvPKh?3=`2d+is%9eJ39X!TnLB>svtp96kKPQU*f0Anm=Zy@yIHx+woU}7f96#`@HR7^sE&O~0q#vX77X;+?!U#}3Q#Vof{=gYt0HmaVe#Z;q8ZRj)O8vqRhVRmew7e8iDm&-1%UKCOVG#(3s+AVo2SuM9aJYg0fW%t&-@koMb*OfxsuiFqZtMP;yb5<fU$Q1HPyGRVzh7)7Ygiz`%V^N}vt)PXDZ<DEKpqW_{Y{auaFS`^!=8M0E8yH@r@9QjHOIdkSTBtHF-}sktanLAFVYcl<SrN0ubvW=Ap&RbAYkb=-4u^M*A)pR2U^MnusUnWOMFyZSJdqJ2c2Ch^_6Wg;3xCZH`%k<i#d@ImWjk$89XEt><>mE!+mc8YywH*4rPj#@g~=n;?o;FJdUxguBR%cwGoYK4SWe|(lSw`&2r}YuAf;`!E$PkPDeMJ&t$Mu<W3d2J&ZFiJ8)`Ob^6<gqN^hAzkJ?<1MP8NI%C}aSCFWEj8o=~vDLRKf*mf+nbmEiI`HI`T5+(;$g792WbMKes*{IDrqOA77TP{QAHrr0(D)EbB|q296#D<>39|BIoMKWEjQ-EJ)>&Gph^iF`(zVIX?^0n#);b)99CmQur`Ep~W&{Q}$Zp-=NPyN)n(%hzESKwkiE+-v6itn2m6CsY5W!YrJQSavaa?zZ2R(UQIHMnFnV2|v>2+k=#{38W=NK}#ZPh_~_H15*QxCO9W9vuL)7E$0Gcg#Z|70JQ;x#Js4qayl#P2+BPoKkwvYJMhOgOwldD^g1ZT$L~gQnuQMC6EX@2LIqBwv5};j}az6nE0WoT1%ETv_J7l9qpbHlz~m@9l)?W?{3#cl4}lxdCQQ#a{vhVJBt83;DY&0!^VzpAaYgBdUpC<UJkLRx&nQ+bRm&n>$jo1}>&o`%PoIKm+>pjowZ^GY}tJ8X(!wNpf&?S+)O&mxGNE+9<c3Dsb@EMH{`nXfuf1(b*N1cn!wN7`l(k@`RBJB~fj|LO0fmDXn6nvu2IFr0XpkH8aRmsa@^I2Arud0S*#PUf%@BU*)-^K28))y)M!DyFO{3<3xn=nMWdW1`3mVF4lvF_(bH!hyBMT!Lj)+AzSB~l)}88Ops>ol+e<|-}VrFb+5Ew(EtDdS#`+;>e~2I00EvB#y0=}1B07+vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())

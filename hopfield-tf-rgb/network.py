import numpy as np
import math
import tensorflow as tf



class Network(object):

    def __init__(self):
        self.sess = tf.Session()

    def construct(self, logdir, visualization_names):



        # visualization

        self.image_matrix = tf.placeholder(tf.float32, (None, None, None), name="image_matrix")
        #self.is_rgb = tf.placeholder(tf.bool, (), name="is_rgb")

        with tf.contrib.summary.create_file_writer(logdir, flush_millis=1 * 1000).as_default(),\
                tf.contrib.summary.always_record_summaries():

            image_matrix_reshaped = tf.reshape(self.image_matrix, shape=[1, tf.shape(self.image_matrix)[0], tf.shape(self.image_matrix)[1], tf.shape(self.image_matrix)[2]])

            self.visualizations = {}
            for vis_name in visualization_names:
                self.visualizations[vis_name] = [
                    tf.contrib.summary.image(("" if "/" in vis_name else "visualization/") + vis_name, image_matrix_reshaped)
                ]
                #self.visualizations[vis_name] = tf.Print(self.visualizations[vis_name], [], f"Vizualization {vis_name}...")


            self.sess.run(tf.global_variables_initializer())
            tf.contrib.summary.initialize(session=self.sess)

    def visualize_image(self, image_matrix, name):
        if image_matrix.ndim == 2:
            image_matrix = np.reshape(image_matrix, (image_matrix.shape[0], image_matrix.shape[1], 1))

        #print(image_matrix.shape)

        self.sess.run(self.visualizations[name], {self.image_matrix: image_matrix})



if __name__ == "__main__":
    net = Network()
    net.construct()
    cube = net.compute_cube(5)
    print(f"cube of 5 = {cube}")
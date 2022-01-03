import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse 
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Conv2D, Dense,MaxPooling2D, Flatten, Dropout
from data_utils import get_data 


target_shape = (200, 200)


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])

    plt.show()

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.01):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

def get_embedding ():
    #base_cnn = resnet.ResNet50(weights="imagenet", input_shape=target_shape + (3,), include_top=False)

    input_layer = Input(shape=(200,200,3))
    conv1 = Conv2D(32, kernel_size=(4, 4), activation="relu") (input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2)) (conv1)
    conv2  = Conv2D(32, kernel_size=(3, 3), activation="relu") (maxpool1)
    maxpool2  = MaxPooling2D(pool_size=(2, 2)) (conv2)
    flt = Flatten()  (maxpool2)
    dropout = Dropout(0.5) (flt)
    output = Dense(128, activation="softmax") (dropout)

    embedding = Model(input_layer, output, name="Embedding")

    print("Embedding Model:\n",embedding.summary())

    return embedding 
 

def get_siamese_network ():
    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    embedding = get_embedding ()

    distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
    )

    siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return  siamese_network


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-data-dir',help='Input data dir')
    parser.add_argument('-o','--output-data-dir',help='Output data dir')

    args = parser.parse_args()

    os.makedirs (args.output_data_dir,exist_ok=True)

    train_dataset, val_dataset = get_data (args)

    visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

    siamese_network =  get_siamese_network()

    print(siamese_network.summary())

    model_checkpoint = ModelCheckpoint(args.output_data_dir + os.sep + 'model.hdf5',
          monitor='val_loss', save_best_only=True, period=1)

    csv_logger = CSVLogger(args.output_data_dir + os.sep + 'history.csv')

    tensorboard = TensorBoard(log_dir = args.output_data_dir + os.sep + "tensorboard",
                histogram_freq = 10,
                write_graph = True,
                write_grads = False,
                write_images = False,
                embeddings_freq = 0,
                embeddings_layer_names = None,
                embeddings_metadata = None,
                embeddings_data = None)



    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001), metrics=["accuracy"])
    siamese_model.fit(train_dataset, epochs=20, validation_data=val_dataset,\
       callbacks= [csv_logger,  tensorboard])


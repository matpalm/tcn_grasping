
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import data

class NormaliseLayer(Layer):
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=1)

class TripletLoss(object):
    def __init__(self, embeddings, embedding_dim, margin):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.margin = margin

    def per_element_hinge_loss(self):
        # slice out anchor, positive and negative embeddings
        embeddings = tf.reshape(self.embeddings, (-1, 3, self.embedding_dim))
        anchor_embeddings = embeddings[:,0]    # (B, E)
        positive_embeddings = embeddings[:,1]  # (B, E)
        negative_embeddings = embeddings[:,2]  # (B, E)
        # calculate distance from anchor to positive and anchor to negative
        dist_a_p = tf.norm(anchor_embeddings - positive_embeddings, axis=1)  # (B)
        dist_a_n = tf.norm(anchor_embeddings - negative_embeddings, axis=1)  # (B)
        # check margin constraint and average over batch
        constraint = dist_a_p - dist_a_n + self.margin                       # (B)
        self.per_element_hinge_loss_op = tf.maximum(0.0, constraint)         # (B)
        return self.per_element_hinge_loss_op

    def triplet_loss(self):
        return tf.reduce_mean(self.per_element_hinge_loss())


def construct_model(embedding_dim, learning_rate=None, margin=None):

    inputs = Input(shape=(data.H, data.W, 3), name='inputs')

    conv = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    conv = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    conv = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    conv = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    conv = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(conv)

    mlp = Flatten()(conv)
    mlp = Dropout(rate=0.5)(mlp)
    mlp = Dense(units=64, activation='relu')(mlp)

    embeddings = Dense(units=embedding_dim, activation=None, name='embedding')(mlp)
    embeddings = NormaliseLayer()(embeddings)

    model = Model(inputs=inputs, outputs=embeddings)
    print(model.summary())

    # use setting of learning_rate (and margin) to denote if
    # we should compile also
    if learning_rate is None:
        # inference only
        return model
    else:
        assert margin is not None
        # use custom loss function since we have no y_true
        # (wrap it in an object as a cleaner way of tracking the internal ops we'll probe)
        loss_fn = TripletLoss(embeddings, embedding_dim, margin)
        model.add_loss(loss_fn.triplet_loss())
        model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=None)
        # for the training model we'll use the model as well as the inputs
        # & the loss_fn so we can feed directly from inputs -> elements
        # in loss for summaries.
        return model, inputs, loss_fn

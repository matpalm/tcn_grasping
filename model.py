
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import data

class NormaliseLayer(Layer):
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=1)

def construct_model(embedding_dim):

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
    return model

class TripletLoss(object):
    def __init__(self, embedding_dim, margin):
        self.embedding_dim = embedding_dim
        self.margin = margin

    def per_element_hinge_loss(self, y_pred):
        # slice out anchor, positive and negative embeddings
        embeddings = tf.reshape(y_pred, (-1, 3, self.embedding_dim))
        anchor_embeddings = embeddings[:,0]    # (B, E)
        positive_embeddings = embeddings[:,1]  # (B, E)
        negative_embeddings = embeddings[:,2]  # (B, E)
        # calculate distance from anchor to positive and anchor to negative
        dist_a_p = tf.norm(anchor_embeddings - positive_embeddings, axis=1)  # (B)
        dist_a_n = tf.norm(anchor_embeddings - negative_embeddings, axis=1)  # (B)
        # check margin constraint and average over batch
        constraint = dist_a_p - dist_a_n + self.margin                       # (B)
        return tf.maximum(0.0, constraint)                                   # (B)

    def triplet_loss(self, _y_true, y_pred):
        return tf.reduce_mean(self.per_element_hinge_loss(y_pred))

def compile(model, embedding_dim, learning_rate, margin):
    loss_fn = TripletLoss(embedding_dim, margin)
    model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                  loss=loss_fn.triplet_loss)
    return model, loss_fn


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
    conv = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    conv = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    conv = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    conv = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(conv)

    mlp = Flatten()(conv)
    mlp = Dropout(rate=0.5)(mlp)
    mlp = Dense(units=64, activation='relu')(mlp)

    embeddings = Dense(units=embedding_dim, activation=None, name='embedding')(mlp)
    embeddings = NormaliseLayer()(embeddings)
    
    model = Model(inputs=inputs, outputs=embeddings)
    print(model.summary())
    return model

def compile(model, embedding_dim, learning_rate, margin):

    # TODO: add margin back
    def triplet_loss(_y_true, y_pred):
        print("y_pred", y_pred)
        # slice out anchor, positive and negative embeddings
        embeddings = tf.reshape(y_pred, (-1, 3, embedding_dim))
        anchor_embeddings = embeddings[:,0]    # (B, E)
        positive_embeddings = embeddings[:,1]  # (B, E)
        negative_embeddings = embeddings[:,2]  # (B, E)
        # calculate distance from anchor to positive and anchor to negative
        dist_a_p = tf.norm(anchor_embeddings - positive_embeddings, axis=1)  # (B)
        dist_a_n = tf.norm(anchor_embeddings - negative_embeddings, axis=1)  # (B)
        # check margin constraint and average over batch
        constraint = dist_a_p - dist_a_n + margin                            # (B)
        per_element_hinge_loss = tf.maximum(0.0, constraint)                 # (B)
        batch_hinge_loss = tf.reduce_mean(per_element_hinge_loss)            # (1)
        return batch_hinge_loss

    model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                  loss=triplet_loss)    
    
    return model
        

    

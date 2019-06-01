#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import argparse
import data
import model as m
import os
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs')
parser.add_argument('--batch-size', type=int, default=16, help="note: effective batch is x3")
parser.add_argument('--embedding-dim', type=int, default=64, help="image embedding dim")
parser.add_argument('--learning-rate', type=float, default=1e-3, help="learning rate for adam")
parser.add_argument('--margin', type=float, default=1e-3, help="hinge loss margin")
parser.add_argument('--negative-frame-range', type=int, default=None,
                    help="select negative +/- this value; if None use entire range")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--steps-per-epoch', type=int, default=20)
parser.add_argument('--run', type=str, default='.',
                    help='run name to use as postfix for model saving, tb output')
parser.add_argument('--model-input', type=str, default=None,
                    help='if set, load weights from this model file')

opts = parser.parse_args()
print(opts)

u.ensure_dir_exists("runs/%s" % opts.run)

examples = data.a_p_n_iterator(batch_size=opts.batch_size,
                               img_dir=opts.img_dir,
                               negative_frame_range=opts.negative_frame_range)

model, inputs, loss_fn = m.construct_model(embedding_dim=opts.embedding_dim,
                                           initial_model=opts.model_input,
                                           learning_rate=opts.learning_rate,
                                           margin=opts.margin)

class NumZeroLossCB(callbacks.Callback):

    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.examples = (data.a_p_n_iterator(batch_size=self.batch_size,
                                             img_dir=opts.img_dir,
                                             negative_frame_range=opts.negative_frame_range).
                         make_one_shot_iterator().get_next())
        self.summary_writer = tf.summary.FileWriter("tb/%s" % opts.run)
        self.loss_histo = tf.summary.histogram("batch_loss_histo", loss_fn.per_element_hinge_loss_op)

    def on_epoch_end(self, epoch, logs):
        # TODO: how do we just use the models iterator here? don't care
        # that it's "wastes" examples doing this eval, it's all generator
        # based anyways...
        next_egs = self.sess.run(self.examples)
        sess = tf.keras.backend.get_session()
        per_elem_loss, pre_margin_loss, loss_histo = sess.run([loss_fn.per_element_hinge_loss_op,
                                                               loss_fn.pre_margin_loss,
                                                               self.loss_histo],
                                                              feed_dict={inputs: next_egs})
        percentage_non_zero = np.count_nonzero(per_elem_loss) / self.batch_size
        summary_values = [tf.Summary.Value(tag='percentage_batch_non_zero_loss',
                                           simple_value=percentage_non_zero),
                          tf.Summary.Value(tag='mean_batch_loss',
                                           simple_value=np.mean(per_elem_loss)),
                          tf.Summary.Value(tag='mean_pre_margin_loss',
                                           simple_value=np.mean(pre_margin_loss))]

        self.summary_writer.add_summary(tf.Summary(value=summary_values), epoch)
        self.summary_writer.add_summary(loss_histo)
        self.summary_writer.flush()

callbacks = [callbacks.ModelCheckpoint(filepath="runs/%s/model.{epoch}.hdf5" % opts.run),
             callbacks.TensorBoard(log_dir="tb/%s" % opts.run),
             NumZeroLossCB()]
model.fit(examples,
          epochs=opts.epochs,
          verbose=1,
          steps_per_epoch=opts.steps_per_epoch,
          callbacks=callbacks)

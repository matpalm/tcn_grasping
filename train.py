#!/usr/bin/env python3

from tensorflow.keras import callbacks
import argparse
import data
import model as m
import numpy as np
import os
import tensorflow as tf
import triplet_selection
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs')
parser.add_argument('--batch-size', type=int, default=16, help="note: effective batch is x3")
parser.add_argument('--embedding-dim', type=int, default=64, help="image embedding dim")
parser.add_argument('--learning-rate', type=float, default=1e-3, help="learning rate for adam")
parser.add_argument('--margin', type=float, default=1e-3, help="hinge loss margin")
parser.add_argument('--negative-frame-range', type=int, default=None,
                    help="select negative +/- this value; if None use entire range. only valid"
                         " for --negative-selection-mode=ranged_frame")
parser.add_argument('--negative-selection-mode', type=str, default='random_frame_random_run',
                    help='one of [random_frame_random_run, random_frame_same_run]')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--steps-per-epoch', type=int, default=20)
parser.add_argument('--run', type=str, default='.',
                    help='run name to use as postfix for model saving, tb output')
parser.add_argument('--model-input', type=str, default=None,
                    help='if set, load weights from this model file')

opts = parser.parse_args()
print(opts)

u.ensure_dir_exists("runs/%s" % opts.run)

triplet_selector = triplet_selection.TripletSelection(opts.img_dir,
                                                      opts.negative_frame_range,
                                                      opts.negative_selection_mode)

examples = data.a_p_n_iterator(opts.batch_size, triplet_selector)

model, inputs, loss_fn = m.construct_model(opts.embedding_dim,
                                           opts.model_input,
                                           opts.learning_rate,
                                           opts.margin)

class NumZeroLossCB(callbacks.Callback):

    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.examples = (data.a_p_n_iterator(self.batch_size, triplet_selector)
                         .make_one_shot_iterator().get_next())
        self.summary_writer = tf.summary.FileWriter("tb/%s" % opts.run)

    def on_epoch_end(self, epoch, logs):
        # TODO: how do we just use the models iterator here? don't care
        # that it's "wastes" examples doing this eval, it's all generator
        # based anyways...
        next_egs = self.sess.run(self.examples)
        sess = tf.keras.backend.get_session()
        summaries = sess.run(loss_fn.summaries, feed_dict={inputs: next_egs})
#        percentage_non_zero = np.count_nonzero(per_elem_loss) / self.batch_size
        # log stats
        for summary in summaries:
            self.summary_writer.add_summary(summary, global_step=epoch)
        self.summary_writer.flush()

callbacks = [callbacks.ModelCheckpoint(filepath="runs/%s/model.{epoch}.hdf5" % opts.run),
             callbacks.TensorBoard(log_dir="tb/%s" % opts.run),
             callbacks.TerminateOnNaN(),
             NumZeroLossCB()]
model.fit(examples,
          epochs=opts.epochs,
          verbose=1,
          steps_per_epoch=opts.steps_per_epoch,
          callbacks=callbacks)

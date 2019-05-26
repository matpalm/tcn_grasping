#!/usr/bin/env python3

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
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--steps-per-epoch', type=int, default=20)
parser.add_argument('--output-dir', type=str, default='.',
                    help='dir to save model.hdf5, tb logs, etc')
opts = parser.parse_args()
print(opts)

u.ensure_dir_exists(opts.output_dir)

examples = data.a_p_n_iterator(batch_size=opts.batch_size,
                               img_dir=opts.img_dir)

model = m.construct_model(embedding_dim=opts.embedding_dim)

m.compile(model,
          embedding_dim=opts.embedding_dim,
          learning_rate=opts.learning_rate,
          margin=opts.margin)

callbacks = [callbacks.ModelCheckpoint(filepath="%s/model.hdf5" % opts.output_dir),
             callbacks.TensorBoard(log_dir="%s/logs" % opts.output_dir)]
model.fit(examples,
          epochs=opts.epochs,
          verbose=1,
          steps_per_epoch=opts.steps_per_epoch,
          callbacks=callbacks)




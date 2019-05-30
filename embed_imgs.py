#!/usr/bin/env python3

import argparse
import model as m
from PIL import Image
import numpy as np
import os
import util as u
import sys
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest', type=str, default='manifest', help='list of files to embed')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--embedding-dim', type=int, default=64, help="image embedding dim")
parser.add_argument('--model-input', type=str, default='model', help='where to load model from')
parser.add_argument('--embeddings-output', type=str, default='embeddings.npy',
                    help='where to write embedding npy')
opts = parser.parse_args()

model = m.construct_model(embedding_dim=opts.embedding_dim)
model.load_weights(opts.model_input)

filenames = list(u.slurp_manifest(opts.manifest))

def filenames_generator():
    for filename in filenames:
        yield filename

def decode_img(img_name):
    return tf.image.decode_jpeg(tf.read_file(img_name))   # (H, W, 3)   uint8

iter = (tf.data.Dataset.from_generator(filenames_generator, output_types=(tf.string))
        .map(decode_img, num_parallel_calls=8)
        .batch(64)
        .prefetch(tf.contrib.data.AUTOTUNE)
        .make_one_shot_iterator()
        .get_next())

# prealloc embeddings and copy them in per batch
embeddings = np.empty((len(filenames), opts.embedding_dim))
e_offset = 0

sess = tf.Session()
while True:
    try:
        imgs = sess.run(iter)
        next_batch = model.predict(imgs)
        embeddings[e_offset:e_offset+len(next_batch)] = next_batch
        e_offset += len(next_batch)
        sys.stdout.write("%d/%d                 \r" % (e_offset, len(filenames)))
    except tf.errors.OutOfRangeError:
        break

np.save(opts.embeddings_output, embeddings)

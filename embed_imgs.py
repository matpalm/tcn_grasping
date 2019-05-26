#!/usr/bin/env python3

import argparse
import model as m
from PIL import Image
import numpy as np
import os
import util as u
import sys

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

# next batch of images to run through
img_batch = []

# TODO: do this as a tf.data pipeline for parallelisation

filenames = list(u.slurp_manifest(opts.manifest))

# prealloc embeddings and copy them in per batch
embeddings = np.empty((len(filenames), opts.embedding_dim))
e_offset = 0

for i, filename in enumerate(filenames):
    # decode image
    pil_img = Image.open(filename)
    img_batch.append(np.array(pil_img))
    # if enoguh images run through batch
    if len(img_batch) == opts.batch_size:
        predictions = model.predict(np.stack(img_batch))
        embeddings[e_offset:e_offset+len(img_batch)] = predictions
        e_offset += len(img_batch)
        img_batch = []
        sys.stdout.write("%d/%d                 \r" % (i, len(filenames)))

# flush final batch
if len(img_batch) > 0:
    predictions = model.predict(np.stack(img_batch))
    embeddings[e_offset:e_offset+len(img_batch)] = predictions
    
# flush embeddings 
print("embeddings.shape", embeddings.shape)
np.save(opts.embeddings_output, embeddings)



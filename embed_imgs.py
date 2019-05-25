#!/usr/bin/env python3

import argparse
import model as m
from PIL import Image
import numpy as np
import os
import util as u

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

# all embeddings
embeddings = []

# next batch of images to run through
img_batch = []

# TODO: do this as a tf.data pipeline for parallelisation

for filename in u.slurp_manifest(opts.manifest):
    # decode image
    pil_img = Image.open(filename)
    img_batch.append(np.array(pil_img))
        
    # if enoguh images run through batch
    if len(img_batch) == opts.batch_size:
        predictions = model.predict(np.stack(img_batch))
        embeddings.append(predictions)
        img_batch = []
        
# stack embeddings into (N, dim) array
embeddings = np.stack(embeddings).reshape((-1, opts.embedding_dim))

# flush final batch
if len(img_batch) > 0:
    predictions = model.predict(np.stack(img_batch))
    embeddings = np.concatenate([embeddings, predictions])
    
# flush embeddings 
print("embeddings.shape", embeddings.shape)
np.save(opts.embeddings_output, embeddings)



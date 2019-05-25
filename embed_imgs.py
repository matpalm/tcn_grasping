#!/usr/bin/env python3

import argparse
import model as m
from PIL import Image
import numpy as np
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--embedding-dim', type=int, default=64, help="image embedding dim")
parser.add_argument('--model-input', type=str, default='model', help='where to load model from')
opts = parser.parse_args()

model = m.construct_model(embedding_dim=opts.embedding_dim)
model.load_weights(opts.model_input)

# all filenames and embeddings
filenames = []
embeddings = []

# next batch of filenames and embeddings to run
filename_batch = []
img_batch = []

for root, dir, imgs in os.walk(opts.img_dir):
    for img_name in imgs:
        
        # decode image
        full_fname = "%s/%s" % (root, img_name)
        filename_batch.append(full_fname)        
        pil_img = Image.open(full_fname)
        img_batch.append(np.array(pil_img))
        
        # if enoguh images run through batch
        if len(img_batch) == opts.batch_size:
            filenames += filename_batch
            predictions = model.predict(np.stack(img_batch))
            embeddings.append(predictions)
            filename_batch = []            
            img_batch = []

# stack embeddings into (N, dim) array
embeddings = np.stack(embeddings).reshape((-1, opts.embedding_dim))

# flush final batch
if len(img_batch) > 0:
    filenames += filename_batch
    predictions = model.predict(np.stack(img_batch))
    embeddings = np.concatenate([embeddings, predictions])

assert len(embeddings) == len(filenames)
    
# flush filenames as json dict
filename_dict = {i: f for i, f in enumerate(filenames)}
import json
with open("filenames.json", "w") as f:
    f.write(json.dumps(filename_dict))

# flush embeddings 
print("embeddings.shape", embeddings.shape)
np.save("embeddings", embeddings)



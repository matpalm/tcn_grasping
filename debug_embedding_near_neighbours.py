#!/usr/bin/env python3

import argparse
import numpy as np
import json
from data import H, W
from PIL import Image, ImageDraw
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filenames-input', type=str, default='filenames.json', 
                    help='where to read filenames json')
parser.add_argument('--embeddings-input', type=str, default='embeddings.npy',
                    help='where to read embedding npy')
opts = parser.parse_args()

fnames = json.loads(open(opts.filenames_input).read())
fnames = {int(k): v for k, v in fnames.items()}
e = np.load(opts.embeddings_input)
assert len(fnames) ==  e.shape[0]

# pick five random idxs for src images
random_idxs = np.random.randint(0, e.shape[0], size=5)

# calculate sims between these five and all others.
all_sims = np.dot(e, e[random_idxs].T)

# calc top 5 near neighbours for each of the the 5
# note: this is reversed, and last (nearest) is itself.
top_5 = np.argsort(all_sims.T)[:,-5:]

# paste near neighbours into collage
BW = 5  # border_width
collage = Image.new('RGB', (5*(W+BW), 5*(H+BW)), (128,128,128))
for row_idx, row_entry in enumerate(random_idxs):
    # paste src image in left column
    print("!", row_idx, row_entry, fnames[row_entry])
    img = u.load_image_with_caption(fnames[row_entry])
    collage.paste(img, (0, (H+BW)*row_idx))
    # paste near neighbours in next 4 columns
    top_4 = reversed(top_5[row_idx][:-1])
    for nn_idx, nn_entry in enumerate(top_4):
        print("!!", nn_idx, nn_entry, fnames[nn_entry])
        img = u.load_image_with_caption(fnames[nn_entry])
        collage.paste(img, ((W+BW)*(nn_idx+1), (H+BW)*row_idx))
collage.show()        

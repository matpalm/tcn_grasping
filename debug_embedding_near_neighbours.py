#!/usr/bin/env python3

import argparse
import numpy as np
from data import H, W
from PIL import Image, ImageDraw
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-a', type=str, default='manifest',
                    help='filenames corresponding to embeddings-a')
parser.add_argument('--embeddings-a', type=str, default='embeddings.npy',
                    help='where to read embedding npy for first set')
parser.add_argument('--manifest-b', type=str, default=None,
                    help='filenames corresponding to embeddings-b. if None use embeddings-a')
parser.add_argument('--embeddings-b', type=str, default=None,
                    help='where to read embedding npy for second set. If None use first set.')
#parser.add_argument('--num-sample', type=int, default=None,
#                    help='if set, only check a sample of first set to compare')
opts = parser.parse_args()

fnames_a = u.slurp_manifest_as_idx_to_name_dict(opts.manifest_a)
e_a = np.load(opts.embeddings_a)
assert len(fnames_a) ==  e_a.shape[0]

if opts.manifest_b is None:
    fnames_b = fname_a
    e_b = e_a
else: 
    fnames_b = u.slurp_manifest_as_idx_to_name_dict(opts.manifest_b)
    e_b = np.load(opts.embeddings_b)
    assert len(fnames_b) ==  e_b.shape[0]

# pick five random idxs for src images
random_idxs = np.random.randint(0, e_a.shape[0], size=5)

# calculate sims between these five and all others.
sims = np.dot(e_a[random_idxs], e_b.T)

# calc top 5 near neighbours for each of the the 5
# note: this is reversed, and last (nearest) is itself.
top_5 = np.argsort(sims.T)[:,-5:]

# paste near neighbours into collage
BW = 5  # border_width
collage = Image.new('RGB', (5*(W+BW), 5*(H+BW)), (128,128,128))
for row_idx, row_entry in enumerate(random_idxs):
    # paste src image in left column
    print("!", row_idx, row_entry, fnames_a[row_entry])
    img = u.load_image_with_caption(fnames_a[row_entry])
    collage.paste(img, (0, (H+BW)*row_idx))
    # paste near neighbours in next 4 columns
    top_4 = reversed(top_5[row_idx][:-1])
    for nn_idx, nn_entry in enumerate(top_4):
        print("!!", nn_idx, nn_entry, fnames_b[nn_entry], sims[row_idx][nn_idx])
        img = u.load_image_with_caption(fnames_b[nn_entry])
        collage.paste(img, ((W+BW)*(nn_idx+1), (H+BW)*row_idx))
collage.show()        

#!/usr/bin/env python3

import argparse
import os
from PIL import Image
import random
from data import H, W

# show a 5x5 collage of random images

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs', help='root dir for imgs')
opts = parser.parse_args()

imgs = []
for root, dirs, files in os.walk(opts.img_dir):
    imgs += ["%s/%s" % (root, f) for f in files]
print(imgs)
N = 5
BW = 5  # pixel border width
collage = Image.new('RGB', ((W+BW)*N, (H+BW)*N), (128,128,128))
for r in range(N):
    for c in range(N):
        img = Image.open(random.choice(imgs))
        collage.paste(img, ((W+BW)*r, (H+BW)*c))
collage.show()

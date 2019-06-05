#!/usr/bin/env python3

from PIL import Image, ImageDraw
from data import H, W
import util as u
import sys

BW = 5  # pixel buffer
N = None
collage = None
for i, line in enumerate(sys.stdin):
    cols = line.strip().split(" ")
    print("/t".join(cols))
    if N is None:
        N = len(cols)
    else:
        assert N == len(cols)
    if collage is None:
        collage = Image.new('RGB', ((W+BW)*N, H), (128,128,128))
    for j, img in enumerate(cols):
        collage.paste(u.load_image_with_caption(img), ((W+BW)*j,0))
    collage.save("stitched_%06d.png" % i)

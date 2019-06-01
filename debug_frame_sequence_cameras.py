#!/usr/bin/env python3

from PIL import Image
from data import H, W
import argparse
import os
import random
import util as u

# show a collage of random images

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs')
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--initial-frame', type=int, default=1)
parser.add_argument('--num-frames', type=int, default=5)
parser.add_argument('--cameras', type=str, default=None, help='comma sep')
opts = parser.parse_args()

camera_ids = list(map(int, opts.cameras.split(",")))
assert len(camera_ids) > 0

num_cols = opts.num_frames
num_rows = len(camera_ids)

collage = Image.new('RGB', (W*num_cols, H*num_rows), (0,0,0))

for frame_offset in range(opts.num_frames):
    for row, camera_id in enumerate(camera_ids):
        fname = "/".join([opts.img_dir,
                          u.run_dir_format(opts.run),
                          u.camera_dir_format(camera_id),
                          u.frame_filename_format(opts.initial_frame + frame_offset)
                         ])
        img = Image.open(fname)
        collage.paste(img, (W*frame_offset, H*row))

collage.show()

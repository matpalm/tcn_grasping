#!/usr/bin/env python3

import argparse
import os
from PIL import Image
import random

# show a 5x5 collage of random images

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

collage = Image.new('RGB', (320*num_cols, 240*num_rows), (0,0,0))

for frame_offset in range(opts.num_frames):
    for row, camera_id in enumerate(camera_ids):
        fname = "%s/c%02d/r%02d/f%03d.png" % (opts.img_dir, camera_id, opts.run,
                                              opts.initial_frame + frame_offset)
        img = Image.open(fname)
        collage.paste(img, (320*frame_offset, 240*row))
        
collage.show()

    
    

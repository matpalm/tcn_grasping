#!/usr/bin/env python3

import tensorflow as tf
from data import a_p_n_iterator, H, W
from PIL import Image, ImageDraw
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs')
parser.add_argument('--negative-frame-range', type=int, default=None,
                    help="select negative +/- this value; if None use entire range")
opts = parser.parse_args()

sess = tf.Session()
apn = a_p_n_iterator(batch_size=4,
                     img_dir=opts.img_dir,
                     negative_frame_range=opts.negative_frame_range)
apn = apn.make_one_shot_iterator().get_next()
examples = sess.run(apn)  #
examples = examples.reshape(4, 3, H, W, 3)  # (batch, APN, H, W, RGB)
print(examples.shape)

BW = 5  # border_width
collage = Image.new('RGB', (4*(W+BW), 3*(H+BW)), (255,255,255))
for i in range(4):
    collage.paste(Image.fromarray(examples[i][0]), (i*(W+BW), 0*(H+BW)))  # anchor
    collage.paste(Image.fromarray(examples[i][1]), (i*(W+BW), 1*(H+BW)))  # positive
    collage.paste(Image.fromarray(examples[i][2]), (i*(W+BW), 2*(H+BW)))  # negative
collage.show()

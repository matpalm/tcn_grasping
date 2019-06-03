#!/usr/bin/env python3

import tensorflow as tf
from data import a_p_n_iterator, H, W
from PIL import Image, ImageDraw
import numpy as np
import argparse
import triplet_selection

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs')
parser.add_argument('--negative-frame-range', type=int, default=None,
                    help="select negative +/- this value; if None use entire range")
parser.add_argument('--negative-selection-mode', type=str, default='random_frame_random_run',
                    help='one of [random_frame_random_run, random_frame_same_run, ranged_frame]')
opts = parser.parse_args()

sess = tf.Session()
triplets = triplet_selection.TripletSelection(img_dir,
                                              negative_frame_range,
                                              negative_selection_mode)
apn = a_p_n_iterator(batch_size=4, triplet_selector=triplets)
apn = apn.make_one_shot_iterator().get_next()
examples = sess.run(apn)
examples = examples.reshape(4, 3, H, W, 3)          # (batch, APN, H, W, RGB) (-1., 1.)
print("examples min", np.min(examples), "max", np.max(examples), "shape", examples.shape)
examples = (((examples+1)/2)*255).astype(np.uint8)  # (-1., 1.) => (0., 2.) => (0., 1.) => (0, 255)

BW = 5  # border_width
collage = Image.new('RGB', (4*(W+BW), 3*(H+BW)), (255,255,255))
for i in range(4):
    collage.paste(Image.fromarray(examples[i][0]), (i*(W+BW), 0*(H+BW)))  # anchor
    collage.paste(Image.fromarray(examples[i][1]), (i*(W+BW), 1*(H+BW)))  # positive
    collage.paste(Image.fromarray(examples[i][2]), (i*(W+BW), 2*(H+BW)))  # negative
collage.show()

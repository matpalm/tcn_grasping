#!/usr/bin/env python3

from PIL import Image, ImageDraw
import os
from data import H, W
import numpy as np
import util as u

# TODO: add opts

# generate synthetic toy data where each frame is it's own colour.
for frame_num, colour in enumerate(['#ff0000', '#ffff00', '#00ff00',
                                    '#00ffff', '#0000ff', '#ff00ff']):    
    for camera_id in range(10):
        output_dir = "imgs/c%02d/r01/" % camera_id
        u.ensure_dir_exists(output_dir)

        img = Image.new('RGB', (W, H), (0, 0, 0))
        canvas = ImageDraw.Draw(img)
        
        # choose rectangle of min size
        rectangle_area = 0
        while rectangle_area < 1000:
            x0, x1 = np.random.randint(0, W, size=2)
            y0, y1 = np.random.randint(0, H, size=2)
            rectangle_area = np.abs((x1-x0) * (y1-y0))
            print("frame_num", frame_num, "rectangle_area", rectangle_area)
        canvas.rectangle([x0,y0,x1,y1], fill=colour)

        img.save("%s/f%03d.png" % (output_dir, frame_num))
       
    

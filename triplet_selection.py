import os
import re
import random

class TripletSelection(object):
    
    def __init__(self, img_dir):
        # use root dir to determine number of cameras
        entries = os.listdir(img_dir)
        for entry in entries:
            if not re.match("c\d\d", entry):
                raise Exception("unexpected entry? [%s]" % entries)
        self.num_cameras = len(entries)

        # collect list of all other run/frame steps
        # use c00 as canonical reference
        self.img_dir = img_dir
        self.frames = []
        for run in os.listdir("%s/c00" % img_dir):
            for frame in os.listdir("%s/c00/%s" % (img_dir, run)):
                self.frames.append("%s/%s" % (run, frame))

    def random_triple(self):
        # anchor is a random frame from a random camera
        anchor_camera = random.randint(0, self.num_cameras-1)
        anchor_frame = random.choice(self.frames)
        anchor_example = "%s/c%02d/%s" % (self.img_dir, anchor_camera, anchor_frame)
        
        # positive is another camera view for the same frame
        positive_camera = anchor_camera
        while positive_camera == anchor_camera:
            positive_camera = random.randint(0, self.num_cameras-1)
        positive_example = "%s/c%02d/%s" % (self.img_dir, positive_camera, anchor_frame)
                
        # negative is a frame from same view, at another time
        negative_frame = anchor_frame
        while negative_frame == anchor_frame:
            negative_frame = random.choice(self.frames)
        negative_example = "%s/c%02d/%s" % (self.img_dir, anchor_camera, negative_frame)

        return anchor_example, positive_example, negative_example

    def random_triples(self):
        while True:
            yield self.random_triple()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img-dir', type=str, default='imgs')
    opts = parser.parse_args()
    ts = TripletSelection(opts.img_dir)
    for i, t in enumerate(ts.random_triples()):
        print(t)
        if i > 10:
            break


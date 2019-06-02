import os
import re
import random
import util as u

class TripletSelection(object):

    def __init__(self, img_dir, negative_frame_range=None):
        self.img_dir = img_dir
        assert negative_frame_range is None or negative_frame_range >= 1
        self.negative_frame_range = negative_frame_range

        # use root dir to determine number of runs
        entries = os.listdir(img_dir)
        for entry in entries:
            if not re.match("^r\d\d\d$", entry):
                raise Exception("unexpected run entry? [%s]" % entries)
        self.num_runs = len(entries)

        # collect size of cameras from first run (assuming same number of all runs!)
        first_run_dir = "%s/%s" % (img_dir, u.run_dir_format(0))
        entries = os.listdir(first_run_dir)
        for entry in entries:
            if not re.match("^c\d\d\d$", entry):
                raise Exception("unexpected camera entry? [%s]" % entries)
        self.num_cameras = len(entries)

        # collect number of frames per camera from first run, first camera
        # again, assuming same number across all runs and cameras
        first_run_first_camera_dir = "%s/%s/%s" % (img_dir,
                                                   u.run_dir_format(0),
                                                   u.camera_dir_format(0))
        self.num_frames = len(os.listdir(first_run_first_camera_dir))

        print("TripletSelection #runs=%d #cameras=%d #frames=%d" % (self.num_runs, self.num_cameras, self.num_frames))

    def random_triple(self):
        # anchor is a random frame from a random camera
        anchor_run = random.randint(0, self.num_runs-1)
        anchor_camera = random.randint(0, self.num_cameras-1)
        anchor_frame = random.randint(0, self.num_frames-1)
        anchor_example = "%s/%s" % (self.img_dir, u.run_camera_frame_filename(anchor_run, anchor_camera, anchor_frame))

        # positive is another camera view for the same run / frame
        positive_camera = anchor_camera
        while positive_camera == anchor_camera:
            positive_camera = random.randint(0, self.num_cameras-1)
        positive_example = "%s/%s" % (self.img_dir, u.run_camera_frame_filename(anchor_run, positive_camera, anchor_frame))

        # negative is a frame from same camera as anchor, but at another (nearby) time
        # TODO: had this as same run, same camera, but things aren't working all of a sudden
        #       so trying this (which was original version)
        negative_run = anchor_run
        negative_frame = anchor_frame
        while negative_run == anchor_run and negative_frame == anchor_frame:
            negative_run = random.randint(0, self.num_runs-1)
            negative_frame = random.randint(0, self.num_frames-1)
#        while negative_frame == anchor_frame or negative_frame < 0 or negative_frame > self.num_frames:
#            if self.negative_frame_range is None:
#                negative_frame = random.randint(0, self.num_frames-1)
#            else:
#                negative_frame = (anchor_frame
#                                  + random.randint(0, 2*self.negative_frame_range)
#                                  - self.negative_frame_range)  # +/- negative_frame_range
        negative_example = "%s/%s" % (self.img_dir, u.run_camera_frame_filename(negative_run, anchor_camera, negative_frame))

        return anchor_example, positive_example, negative_example

    def random_triples(self):
        while True:
            yield self.random_triple()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img-dir', type=str, default=None)
    parser.add_argument('--negative-frame-range', type=int, default=None)
    opts = parser.parse_args()
    assert opts.img_dir is not None
    ts = TripletSelection(opts.img_dir, opts.negative_frame_range)
    for i, t in enumerate(ts.random_triples()):
        print(t)
        if i > 10:
            break

import os
import re
import random
import util as u

class TripletSelection(object):

    def __init__(self, img_dir):
        # use root dir to determine number of cameras
        entries = os.listdir(img_dir)
        for entry in entries:
            if not re.match("^c\d\d$", entry):
                raise Exception("unexpected entry? [%s]" % entries)
        self.num_cameras = len(entries)

        # collect size of runs / frames per run assuming c00 is a
        # canconical reference for other cameras
        self.img_dir = img_dir
        self.num_runs = len(os.listdir("%s/c00" % img_dir))
        self.run_to_num_frames = [0] * self.num_runs
        for run_id in range(self.num_runs):
            num_frames = len(os.listdir("%s/c00/r%02d" % (img_dir, run_id)))
            assert num_frames >= 1
            self.run_to_num_frames[run_id] = num_frames

        print("TripletSelection #cameras=%d run_to_num_frames=%s" % (self.num_cameras,
                                                                     self.run_to_num_frames))

    def random_triple(self):
        # anchor is a random frame from a random camera
        anchor_camera = random.randint(0, self.num_cameras-1)
        anchor_run = random.randint(0, self.num_runs-1)
        anchor_frame = random.randint(0, self.run_to_num_frames[anchor_run])
        anchor_example = "%s/%s" % (self.img_dir, u.camera_img_fname(anchor_camera, anchor_run, anchor_frame))

        # positive is another camera view for the same run / frame
        positive_camera = anchor_camera
        while positive_camera == anchor_camera:
            positive_camera = random.randint(0, self.num_cameras-1)
        positive_example = "%s/%s" % (self.img_dir, u.camera_img_fname(positive_camera, anchor_run, anchor_frame))

        # negative is a frame from same run as anchor, but at another time
        negative_frame = anchor_frame
        while negative_frame == anchor_frame:
            negative_frame = random.randint(0, self.run_to_num_frames[anchor_run])
        negative_example = "%s/%s" % (self.img_dir, u.camera_img_fname(anchor_camera, anchor_run, negative_frame))

        return anchor_example, positive_example, negative_example

    def random_triples(self):
        while True:
            yield self.random_triple()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img-dir', type=str, default=None)
    opts = parser.parse_args()
    assert opts.img_dir is not None
    ts = TripletSelection(opts.img_dir)
    for i, t in enumerate(ts.random_triples()):
        print(t)
        if i > 10:
            break

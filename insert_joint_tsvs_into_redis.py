#!/usr/bin/env python3
import argparse
import os
import re
import redis
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--joint-info-dir', type=str, help='root dir to look for .tsvs')
opts = parser.parse_args()

r = redis.Redis()
inserts = 0
for root, dir, files in os.walk(opts.joint_info_dir):
    m = re.match(".*\/r(\d\d\d)$", root)
    if not m: continue
    run_id = str(int(m.group(1)))  # '001' => '1'
    for file in files:
        m = re.match("c(\d\d\d).tsv", file)
        if not m: continue
        camera_id = str(int(m.group(1)))
        for line in open(root+"/"+file, "r"):
            cols = line.strip().split("\t")
            assert len(cols) == 15, len(cols)
            frame_id = str(int(cols[0]))
            key = "|".join([opts.joint_info_dir, run_id, camera_id, frame_id])
            value = np.array(list(map(float, cols[1:8])))
            r.set(key, value.tobytes())
            inserts += 1

print("#inserts", inserts)

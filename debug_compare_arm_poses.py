#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import random
import re
import redis
import seaborn as sns
import sys

r = redis.Redis()

def lookup(key):
    array_bytes = r.get(key)
    if array_bytes is None:
        raise Exception("key [%s] not in db?" % key)
    return np.frombuffer(array_bytes)

def pose_distance(joint_info_dir, run_a, camera_a, frame_a, run_b, camera_b, frame_b):
    key_a = "|".join(map(str, [joint_info_dir, run_a, camera_a, frame_a]))
    pose_a = lookup(key_a)
    key_b = "|".join(map(str, [joint_info_dir, run_b, camera_b, frame_b]))
    pose_b = lookup(key_b)
    dist = np.linalg.norm(pose_a - pose_b)
    return dist

def rcf(filename):
    m = re.match(".*/r(\d\d\d)/c(\d\d\d)/f(\d\d\d\d).png$", filename)
    return tuple(map(int, m.groups()))


distances_ref_target_a = []
distances_ref_target_b = []
for line in open("stitch.out", "r"):
    c = eval(line.strip())
    for i in range(3):
        assert c[i].startswith("imgs/03_heldout/")
    r0, c0, f0 = rcf(c[0])
    r1, c1, f1 = rcf(c[1])
    r2, c2, f2 = rcf(c[2])
    distances_ref_target_a.append(pose_distance("joint_infos/03_heldout/", r0, c0, f0, r1, c1, f1))
    distances_ref_target_b.append(pose_distance("joint_infos/03_heldout/", r0, c0, f0, r2, c2, f2))

sns.distplot(distances_ref_target_a, label='...target_a')
plt.axvline(np.mean(distances_ref_target_a), c='blue')
sns.distplot(distances_ref_target_b, label='...target_b')
plt.axvline(np.mean(distances_ref_target_b), c='red')

distances = []
for _ in range(1000):
    r1 = random.randint(0, 99)
    c1 = random.randint(0, 2)
    f1 = random.randint(0, 999)
    r2 = random.randint(0, 99)
    c2 = random.randint(0, 2)
    f2 = random.randint(0, 999)
    distances.append(pose_distance("joint_infos/03_heldout/", r1, c1, f1, r2, c2, f2))

#plt.clf()
sns.distplot(distances, label='...random')
plt.axvline(np.mean(distances), c='green')

plt.title("pose distances from reference to ...")
plt.legend()
plt.savefig("blog_imgs/pose_distances_from_ref.png")

#!/usr/bin/env python3
import random
for run_sid in range(100):
    learning_rate = random.choice([1e-5, 1e-4, 1e-3])
    margin = random.choice([0, 1e-4, 1e-3, 1e-2, 0.1])
    run_id = "29_%02d_lr%f_m%f" % (run_sid, learning_rate, margin)
    cmd = "./train.py"
    cmd += " --run %s" % run_id
    cmd += " --img-dir imgs/03"
    cmd += " --learning-rate %f" % learning_rate
    cmd += " --epochs 100"
    cmd += " --steps-per-epoch 100"
    cmd += " --embedding-dim 32"
    cmd += " --margin %f" % margin
    cmd += " >29_%s.out 2>29_%s.err" % (run_sid, run_sid)
    print(cmd)

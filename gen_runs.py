#!/usr/bin/env python3
import random

# margin 0, 0.0001, 0.001, 0.01
# random_frame_random_run, ranged_frame:1000, ranged_frame:100, ranged_frame:10

BASE_RUN_ID = 39
run_sid = 0

print("set -ex")

def combos():
    margin = 0.1
    learning_rate = 1e-4
    embedding_dim = 128
    for negative_strategy in ['ranged_frame:100', 'ranged_frame:10']:
        yield margin, negative_strategy, learning_rate, embedding_dim

for margin, negative_strategy, learning_rate, embedding_dim in combos():
    run_sid += 1
    for sub_run in range(4):
        run_id = "%d_%03d_%d" % (BASE_RUN_ID, run_sid, sub_run)
        cmd = "time ./train.py"
        cmd += " --run %s" % run_id
        cmd += " --img-dir imgs/03"
        cmd += " --learning-rate %s" % learning_rate
        cmd += " --epochs 100"
        cmd += " --steps-per-epoch 200"
        cmd += " --embedding-dim %d" % embedding_dim
        cmd += " --margin %s" % margin
        if negative_strategy == 'random_frame_random_run':
            cmd += " --negative-selection-mode random_frame_random_run"
        else:
            nrf = negative_strategy.replace("ranged_frame:", "")
            cmd += " --negative-selection-mode ranged_frame"
            cmd += " --negative-frame-range %s" % nrf
        cmd += " --model-input runs/38_004_0/model.100.hdf5"
        cmd += " >logs/%s.out 2>logs/%s.err" % (run_id, run_id)
        print(cmd)

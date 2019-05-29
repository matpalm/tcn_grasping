#!/usr/bin/env python3

learning_rate = 1e-3
margin = 0.1
for run in range(16, 19):
    out_dir = "runs/%02d" % run
    cmd = "mkdir %s; ./train.py" % out_dir
    cmd += " --img-dir imgs/02_20c_10o.lores/train"
    cmd += " --embedding-dim 32"
    cmd += " --learning-rate %f" % learning_rate
    cmd += " --margin %f" % margin
    cmd += " --epochs 30"
    cmd += " --steps-per-epoch 1000"
    cmd += " --run %02d" % run
    cmd += " >%s/out 2>%s/err" % (out_dir, out_dir)
    print(cmd)

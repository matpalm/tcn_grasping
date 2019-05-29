#!/usr/bin/env python3

run = 7
for learning_rate in [1e-3, 1e-4]:
    for margin in [0, 0.1, 0.2]:
        out_dir = "runs/%02d" % run
        cmd = "mkdir %s; ./train.py" % out_dir
        cmd += " --img-dir imgs/02_20c_10o.lores/train"
        cmd += " --embedding-dim 32"
        cmd += " --learning-rate %f" % learning_rate
        cmd += " --margin %f" % margin
        cmd += " --epochs 100"
        cmd += " --steps-per-epoch 1000"
        cmd += " --run %02d" % run
        cmd += " >%s/out 2>%s/err" % (out_dir, out_dir)
        print(cmd)
        run += 1

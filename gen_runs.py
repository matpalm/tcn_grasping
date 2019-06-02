#!/usr/bin/env python3

learning_rate = 1e-3
margin = 0.1

run_sid = 1


for learning_rate in [1e-5, 1e-4, 1e-3]:
    for margin in [0, 1e-4, 1e-3, 1e-2, 0.1]:
        for embedding_dim in [16, 32, 64]:
            run_id = "29_%02d_lr%f_m%f_e%d" % (run_sid, learning_rate, margin, embedding_dim)
            cmd = "./train.py"
            cmd += " --run %s" % run_id
            cmd += " --img-dir imgs/03"
            cmd += " --learning-rate %f" % learning_rate
            cmd += " --epochs 600"
            cmd += " --steps-per-epoch 200"
            cmd += " --embedding-dim %d" % embedding_dim
            cmd += " --margin %f" % margin
            cmd += " >29_%s.out 2>29_%s.err" % (run_sid, run_sid)
            print(cmd)
            run_sid += 1

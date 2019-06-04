#!/usr/bin/env python3
import tensorflow as tf
import re
import glob
import os

# what --opts columns to use?
OPTION_KEYS = ("embedding_dim learning_rate margin negative_frame_range"
               " negative_selection_mode".split(" "))

def options_for_run(run_id):
    for line in open("logs/%s.out" % run_id).readlines():
        m = re.match("^Namespace\((.*)\)$", line.strip())
        assert m
        opts = {}
        for opt_str in m.group(1).split(","):
            key, value = opt_str.strip().split("=")
            if key in OPTION_KEYS:
                opts[key] = value
        # return in deterministic order
        return [opts[k] for k in OPTION_KEYS]

def event_files_for_run(run_id):
    return glob.glob("tb/%s/events*" % run_id)

# header
print("\t".join(['exp', 'sample'] + OPTION_KEYS + ['step', 'k', 'v']))

for run in os.listdir('tb'):
    m = re.match("^(.*)_(\d)$", run)
    exp, sample = m.groups()
    opts = options_for_run(run)
    for event_file in event_files_for_run(run):
        for e in tf.train.summary_iterator(event_file):
            # all the meta events have an (implied) step value
            # of zero so ignoring any event with 0 skips these.
            # TODO: dangerous!!
            if e.step == 0: continue
            step = e.step
            key = e.summary.value[0].tag
            value = e.summary.value[0].simple_value
            print("\t".join(map(str, [exp, sample] + opts + [step, key, value])))

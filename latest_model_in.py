#!/usr/bin/env python3
import sys, re, os
model_dir = sys.argv[1]
most_recent_model = -1
for entry in os.listdir(model_dir):
    m = re.match("model\.(\d*)\.hdf5", entry)
    if m:
        model_id = int(m.group(1))
        if model_id > most_recent_model:
            most_recent_model = model_id
if most_recent_model == -1:
    raise Exception("no models is [%s]??" % model_dir)
print("%s/model.%d.hdf5" % (model_dir, most_recent_model))

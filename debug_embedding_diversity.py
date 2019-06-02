#!/usr/bin/env python3
import numpy as np
import sys
embeddings = np.load(sys.argv[1])
print(np.std(embeddings, axis=0))

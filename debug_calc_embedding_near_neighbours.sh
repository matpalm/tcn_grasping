#!/usr/bin/env bash
set -ex

# take snapshop of model (in case still training)
cp runs/$1/model.hdf5 /tmp

# embed images from reference sequence
time ./embed_imgs.py \
--manifest imgs/03/c087.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 64 \
     --embeddings-output runs/$1/03.c087.embeddings.npy

# embed all images from entire test set
#time ./embed_imgs.py \
##--manifest imgs/02_20c_10o.lores/test.manifest \
##--model-input runs/$1/model.hdf5 \
##--embedding-dim 32 \
##     --embeddings-output runs/$1/02_20c_10o.lores.test.embeddings.npy  # ????

# embed images from test set, c18 and c19, each
time ./embed_imgs.py \
--manifest imgs/03/c088.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 64 \
--embeddings-output runs/$1/03.c088.embeddings.npy
time ./embed_imgs.py \
--manifest imgs/03/c089.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 64 \
--embeddings-output runs/$1/03.c089.embeddings.npy

# do near neighbour calcs between (ref, c18) & (ref, c19)
./debug_embedding_near_neighbours.py \
--manifest-a imgs/03/c087.manifest \
--embeddings-a runs/$1/03.c087.embeddings.npy \
--manifest-b imgs/03/c088.manifest \
--embeddings-b runs/$1/03.c088.embeddings.npy > /tmp/c88.nns &
./debug_embedding_near_neighbours.py \
--manifest-a imgs/03/c087.manifest \
--embeddings-a runs/$1/03.c087.embeddings.npy \
--manifest-b imgs/03/c089.manifest \
--embeddings-b runs/$1/03.c089.embeddings.npy > /tmp/c89.nns &
wait

# join into one file, sample and stitch
join /tmp/c88.nns /tmp/c89.nns | awk 'NR%10==0' | ./stitch.py

# make a gif
convert stitch*png runs/$1/near_neighbour_egs.gif

# clean up
rm stitch*png /tmp/{c88,c89}.nns /tmp/model.hdf5

eog runs/$1/near_neighbour_egs.gif

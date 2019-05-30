#!/usr/bin/env bash
set -ex

# take snapshop of model (in case still training)
cp runs/$1/model.hdf5 /tmp/

# embed images from reference sequence
time ./embed_imgs.py \
--manifest imgs/02_20c_10o.lores/c00.r00.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 64 \
--embeddings-output runs/$1/02_20c_10o.lores.c00.r00.embeddings.npy  # (800, 32)

# embed all images from entire test set
##time ./embed_imgs.py \
##--manifest imgs/02_20c_10o.lores/test.manifest \
##--model-input runs/$1/model.hdf5 \
##--embedding-dim 32 \
##     --embeddings-output runs/$1/02_20c_10o.lores.test.embeddings.npy  # ????

# embed images from test set, c18 and c19, each
time ./embed_imgs.py \
--manifest imgs/02_20c_10o.lores/c18.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 64 \
--embeddings-output runs/$1/02_20c_10o.lores.c18.embeddings.npy
time ./embed_imgs.py \
--manifest imgs/02_20c_10o.lores/c19.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 64 \
--embeddings-output runs/$1/02_20c_10o.lores.c19.embeddings.npy

# do near neighbour calcs between (ref, c18) & (ref, c19)
./debug_embedding_near_neighbours.py \
--manifest-a imgs/02_20c_10o.lores/c00.r00.manifest \
--embeddings-a runs/$1/02_20c_10o.lores.c00.r00.embeddings.npy \
--manifest-b imgs/02_20c_10o.lores/c18.manifest \
--embeddings-b runs/$1/02_20c_10o.lores.c18.embeddings.npy > /tmp/c18.nns &
./debug_embedding_near_neighbours.py \
--manifest-a imgs/02_20c_10o.lores/c00.r00.manifest \
--embeddings-a runs/$1/02_20c_10o.lores.c00.r00.embeddings.npy \
--manifest-b imgs/02_20c_10o.lores/c19.manifest \
--embeddings-b runs/$1/02_20c_10o.lores.c19.embeddings.npy > /tmp/c19.nns &
wait

# join into one file, sample and stitch
join /tmp/c18.nns /tmp/c19.nns | awk 'NR%10==0' | ./stitch.py

# make a gif
convert stitch*png runs/$1/near_neighbour_egs.gif

# clean up
rm stitch*png /tmp/{c18,c19}.nns /tmp/model.hdf5

eog runs/$1/near_neighbour_egs.gif

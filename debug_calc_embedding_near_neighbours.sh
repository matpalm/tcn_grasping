#!/usr/bin/env bash
set -ex

# take snapshop of model (in case still training)
cp runs/$1/model.hdf5 /tmp

# embed images from reference sequence
time ./embed_imgs.py \
--manifest source.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 32 \
--embeddings-output /tmp/source.embeddings.npy

# embed images from target_a set
time ./embed_imgs.py \
--manifest target_a.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 32 \
--embeddings-output /tmp/target_a.embeddings.npy

# embed images from target_b set
time ./embed_imgs.py \
--manifest target_b.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim 32 \
--embeddings-output /tmp/target_b.embeddings.npy

# do near neighbour calcs between (source, target_a) & (source, target_b)
./debug_embedding_near_neighbours.py \
--manifest-a source.manifest \
--embeddings-a /tmp/source.embeddings.npy \
--manifest-b target_a.manifest \
--embeddings-b /tmp/target_a.embeddings.npy > /tmp/source_target_a.nns &
./debug_embedding_near_neighbours.py \
--manifest-a source.manifest \
--embeddings-a /tmp/source.embeddings.npy \
--manifest-b target_b.manifest \
--embeddings-b /tmp/target_b.embeddings.npy > /tmp/source_target_b.nns &
wait

# join into one file, sample and stitch
#join /tmp/source_target_[ab].nns | awk 'NR%10==0' | ./stitch.py
join /tmp/source_target_[ab].nns | ./stitch.py

exit

# make a gif
convert stitch*png runs/$1/near_neighbour_egs.gif

# clean up
rm stitch*png /tmp/{source,target_a,target_b}.embeddings.npy /tmp/source_target_[ab].nns /tmp/model.hdf5

# show nn gif
eog runs/$1/near_neighbour_egs.gif

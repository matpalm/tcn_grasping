#!/usr/bin/env bash
set -ex

EMBEDDING_DIM=128

#find imgs/03_heldout/ -type f | sort | gzip > imgs_03.manifest.gz
zcat imgs_03.manifest.gz | grep r000.c000 | awk 'NR%3==0' > source.manifest &
zcat imgs_03.manifest.gz | grep c001 | awk 'NR%5==0' > target_a.manifest &
zcat imgs_03.manifest.gz | grep c002 | awk 'NR%5==0' > target_b.manifest &
wait
wc -l *manifest

# take snapshop of latest model (in case still training)
cp `./latest_model_in.py runs/$1` /tmp/model.hdf5

# embed images from reference sequence
time ./embed_imgs.py \
--manifest source.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim $EMBEDDING_DIM \
--embeddings-output /tmp/source.embeddings.npy

# embed images from target_a set
time ./embed_imgs.py \
--manifest target_a.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim $EMBEDDING_DIM \
--embeddings-output /tmp/target_a.embeddings.npy

# embed images from target_b set
time ./embed_imgs.py \
--manifest target_b.manifest \
--model-input /tmp/model.hdf5 \
--embedding-dim $EMBEDDING_DIM \
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
join /tmp/source_target_[ab].nns > runs/$1/source_target_joined.ssv

exit

# stitch images
./stitch.py < source_target_joined.ssv > stitch.out

# make a gif
convert stitch*png runs/$1/near_neighbour_egs.gif

# clean up
rm stitch*png /tmp/{source,target_a,target_b}.embeddings.npy /tmp/source_target_[ab].nns /tmp/model.hdf5

# show nn gif
eog runs/$1/near_neighbour_egs.gif

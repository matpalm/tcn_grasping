import tensorflow as tf
import triplet_selection

#H, W = 240, 320
H, W = 180, 240
#H, W = 120, 160

NUM_PARALLEL_CALLS = 4

def decode(img_name):
  img = tf.image.decode_jpeg(tf.read_file(img_name))   # (H, W, 3)   uint8
  img = tf.cast(img, tf.float32)
  img = (img / 127.5) - 1.0       # (-1, 1)
  return img

def decode_triple(a, p, n):
  return tf.stack([decode(a), decode(p), decode(n)])  # (3, H, W, 3)

def flatten_apn_into_batch(batched_a_p_n):
  return tf.reshape(batched_a_p_n, (-1, H, W, 3))

def a_p_n_iterator(batch_size, img_dir, negative_frame_range):
  # return batch of examples (batch, 3(apn), h, w, 3(rgb))

  triplets = triplet_selection.TripletSelection(img_dir, negative_frame_range)

  # TODO: from_generator triggers py_func => deprecated
  dataset = tf.data.Dataset.from_generator(triplets.random_triples,
                                           output_types=(tf.string,   # anchor
                                                         tf.string,   # positive
                                                         tf.string))  # negative

  dataset = dataset.map(decode_triple, num_parallel_calls=NUM_PARALLEL_CALLS)

  batched_dataset = dataset.batch(batch_size)
  batched_dataset = batched_dataset.map(flatten_apn_into_batch,
                                        num_parallel_calls=NUM_PARALLEL_CALLS)

  # recall: keras takes iterator, not .get_next())
  # TODO: contrib deprecated
  return batched_dataset.prefetch(tf.contrib.data.AUTOTUNE)

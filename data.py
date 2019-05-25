import tensorflow as tf
import triplet_selection

H, W = 240, 320

def decode(img_name):
  return tf.image.decode_jpeg(tf.read_file(img_name))   # (H, W, 3)   uint8

def decode_triple(a, p, n):
  return tf.stack([decode(a), decode(p), decode(n)])  # (3, H, W, 3)

def add_dummy_label(a_p_n):
  # include a dummy label as part of the iterator since i don't
  # how to weave in the triplet loss under the compile api (which
  # requires loss to have a y_true & y_pred
  return a_p_n, 0

def flatten_apn_into_batch(batched_a_p_n, batched_labels):
  return (tf.reshape(batched_a_p_n, (-1, H, W, 3)),
          batched_labels)

def a_p_n_iterator(batch_size, img_dir):
  # return batch of examples (batch, 3(apn), h, w, 3(rgb))
  
  triplets = triplet_selection.TripletSelection(img_dir)
  
  # TODO: from_generator triggers py_func => deprecated
  dataset = tf.data.Dataset.from_generator(triplets.random_triples,
                                           output_types=(tf.string,   # anchor
                                                         tf.string,   # positive
                                                         tf.string))  # negative
  
  dataset = dataset.map(decode_triple, num_parallel_calls=4)
  dataset = dataset.map(add_dummy_label, num_parallel_calls=4)
  
  batched_dataset = dataset.batch(batch_size)
  batched_dataset = batched_dataset.map(flatten_apn_into_batch, num_parallel_calls=4)

  # recall: keras takes iterator, not .get_next())
  # TODO: contrib deprecated  
  return batched_dataset.prefetch(tf.contrib.data.AUTOTUNE)   


  

  

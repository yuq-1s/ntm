#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys

FILENAME = 'data/sequences.tfrecord'

def get_example(bit_width, seq_length, batch_size):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    record = np.array([np.array([np.random.randint(0, 2, bit_width) \
                                 for _ in range(seq_length)]) \
                       for n in range(batch_size)])
    with tf.python_io.TFRecordWriter(FILENAME) as writer:
        feature = {'sequence': _bytes_feature(tf.compat.as_bytes(
            record.tostring()))}
        writer.write(tf.train.Example(features=tf.train.Features(
            feature=feature)).SerializeToString())

if __name__ == '__main__':
    if len(sys.argv) == 4:
        get_example(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        print("Wrote %d bit_width, %d seq_length, %d batch_size to %s" % \
              (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), FILENAME))
    else:
        print("Usage: %s <bit_width> <seq_length> <batch_size>" % sys.argv[0])

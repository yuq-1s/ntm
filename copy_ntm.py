#! /usr/bin/env python3

import tensorflow as tf
from ntm import NTM

FLAGS = tf.app.flags.FLAGS
# FIXME: put these to a config file
BIT_WIDTH = 8

tf.app.flags.DEFINE_integer('N', 10, "memory size")
tf.app.flags.DEFINE_integer('M', 100, "memory width")
tf.app.flags.DEFINE_boolean('use_lstm', True, "usr lstm or linear controller")

class CopyNTM(NTM):
    def __init__(self, inputs, N, M, use_lstm, params={}):
        padding = tf.zeros_like(inputs)
        # FIXME: just concat this, there must bugs here
        new_inputs = tf.concat([inputs, padding], 1)
        self.labels = tf.concat([padding, inputs], 1)
        super().__init__(new_inputs, N, M, use_lstm, params)

    @property
    def loss(self):
        if hasattr(self, '_loss'):
            return self._loss
        else:
            self._loss = tf.losses.absolute_difference(
                self.outputs, self.labels)
            return self._loss

    @property
    def metrics(self):
        return self.loss

def process_record(record):
    features = tf.parse_single_example(
        record,
        features={'s': tf.FixedLenSequenceFeature([BIT_WIDTH], tf.string,
                                                  allow_missing=True)})
    # TODO: save as bool to compress tfrecord size?
    seq = tf.decode_raw(features['s'], tf.int32)
    seq = tf.cast(features['s'], tf.float32)
    return seq

def load_op(filename):
    return tf.data.TFRecordDataset(filename) \
        .map(process_record) \
        .shuffle(buffer_size=1024) \
        .batch(32) \
        .repeat() \
        .make_one_shot_iterator() \
        .get_next()

def main(_):
    ntm = CopyNTM(load_op('data/sequences.tfrecord'),
                  FLAGS.N,
                  FLAGS.M,
                  FLAGS.use_lstm)

if __name__ == '__main__':
    tf.app.run()

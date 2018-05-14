#! /usr/bin/env python3

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import numpy as np
import random
from ntm import NTM
import dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('N', 30, "memory size")
tf.app.flags.DEFINE_integer('M', 10, "memory width")
tf.app.flags.DEFINE_boolean('use_lstm', True, "usr lstm or linear controller")
tf.app.flags.DEFINE_integer('train_steps', 100, "steps to train")
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('bit_width', 8, 'bits to copy per time step')
tf.app.flags.DEFINE_integer('max_seq_length', 20, 'maximum length of sequence'
                            'to copy')
tf.app.flags.DEFINE_integer('min_seq_length', 10, 'minimum length of sequence'
                            'to copy')
tf.app.flags.DEFINE_integer('random_seed', 42, 'random seed')
tf.app.flags.DEFINE_integer('num_batch', 10, 'number of batches for training')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'learning rate')
tf.app.flags.DEFINE_integer('num_classes', 2, 'number of classes to be copied')
tf.app.flags.DEFINE_integer('steps_per_batch', 8, 'steps to run for each batch')
tf.app.flags.DEFINE_string('dataset', 'data/train.tfrecord', 'path to tfrecord')

TOTAL_TIME_LENGTH = FLAGS.max_seq_length * 2 + 2
TOTAL_BIT_WIDTH = FLAGS.bit_width + 2

class CopyNTM(NTM):
    def __call__(self, inputs, labels, lengths, params):
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths
        return super().model_fn(inputs, labels, lengths, params)

    @property
    def loss(self):
        with tf.variable_scope('loss'):
            if hasattr(self, '_loss'):
                return self._loss
            else:
                mask = tf.not_equal(
                    self.labels, tf.ones_like(self.labels)*FLAGS.num_classes,
                    name='mask')
                raw_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=tf.reshape(
                        self.logits,
                        [FLAGS.batch_size, TOTAL_TIME_LENGTH, TOTAL_BIT_WIDTH,
                        FLAGS.num_classes]),
                    labels=tf.one_hot(self.labels, FLAGS.num_classes))
                self._loss = tf.reduce_sum(tf.boolean_mask(raw_loss, mask)) / \
                    tf.cast(tf.count_nonzero(mask), tf.float32)
                tf.summary.scalar('loss', self._loss)
                return self._loss

    @property
    def metrics(self):
        with tf.variable_scope('metrics'):
            if hasattr(self, '_metrics'):
                return self._metrics
            else:
                mask = tf.not_equal(
                    self.labels, tf.ones_like(self.labels)*FLAGS.num_classes,
                    name='mask')
                predictions = tf.argmax(
                    tf.reshape(
                        self.logits,
                        [FLAGS.batch_size, TOTAL_TIME_LENGTH,
                        TOTAL_BIT_WIDTH, FLAGS.num_classes]),
                    axis=3, name='predictions')
                accuracy = tf.count_nonzero(
                    tf.equal(tf.boolean_mask(self.labels, mask),
                            tf.boolean_mask(predictions, mask))) / \
                    tf.count_nonzero(mask)
                self._metrics = {'accuracy': accuracy}
                tf.summary.scalar('accuracy', accuracy)
                return self._metrics

def main(_):
    tf.logging.set_verbosity('DEBUG')
    params={
        'N': FLAGS.N,
        'M': FLAGS.M,
        'use_lstm': FLAGS.use_lstm,
        'batch_size': FLAGS.batch_size,
        'bit_width': FLAGS.bit_width,
        'learning_rate': FLAGS.learning_rate,
        'num_classes': FLAGS.num_classes}

    ntm = CopyNTM()
    data = dataset.load_op(
        FLAGS.dataset, TOTAL_TIME_LENGTH, TOTAL_BIT_WIDTH, FLAGS.batch_size)
    train_op = ntm(*data, params)

    # FIXME: write summary per step only for debug purpose
    with tf.train.MonitoredTrainingSession(
            save_summaries_steps=1,
            checkpoint_dir='model') as sess:
        for _ in range(FLAGS.train_steps):
            sess.run(train_op)

if __name__ == '__main__':
    tf.app.run()

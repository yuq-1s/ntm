#! /usr/bin/env python3

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import numpy as np
import random
import pathlib

import ntm
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
tf.app.flags.DEFINE_integer('num_batch', 100, 'number of batches for training')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'learning rate')
tf.app.flags.DEFINE_integer('num_classes', 2, 'number of classes to be copied')
tf.app.flags.DEFINE_integer('steps_per_batch', 8, 'steps to run for each batch')
tf.app.flags.DEFINE_string('dataset', 'data/train.tfrecord', 'path to tfrecord')

TOTAL_TIME_LENGTH = FLAGS.max_seq_length * 2 + 2
TOTAL_BIT_WIDTH = FLAGS.bit_width + 2

class CopyNTM(ntm.NTM):
    def __call__(self, inputs, labels, lengths, params):
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths
        self.mask = tf.not_equal(
            self.labels, tf.ones_like(self.labels)*FLAGS.num_classes,
            name='mask')
        self.total_seq_length = tf.cast(tf.count_nonzero(self.mask), tf.float32)
        return super().model_fn(inputs, labels, lengths, params)

    @property
    def loss(self):
        with tf.variable_scope('loss'):
            if hasattr(self, '_loss'):
                return self._loss
            else:
                raw_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=tf.reshape(
                        self.logits,
                        [FLAGS.batch_size, TOTAL_TIME_LENGTH, TOTAL_BIT_WIDTH,
                        FLAGS.num_classes]),
                    labels=tf.one_hot(self.labels, FLAGS.num_classes),
                    name='raw_loss')
                self._loss = tf.div(
                    tf.reduce_sum(
                        tf.boolean_mask(raw_loss, self.mask, name='masked_loss')
                    ), self.total_seq_length, name='loss'
                )
                tf.summary.scalar('loss', self._loss)
                return self._loss

    @property
    def metrics(self):
        with tf.variable_scope('metrics'):
            if hasattr(self, '_metrics'):
                return self._metrics
            else:
                predictions = tf.argmax(
                    tf.reshape(
                        self.logits,
                        [FLAGS.batch_size, TOTAL_TIME_LENGTH,
                        TOTAL_BIT_WIDTH, FLAGS.num_classes]),
                    axis=3, name='predictions')
                accuracy = tf.div(
                    tf.cast(
                        tf.count_nonzero(
                            tf.boolean_mask(
                                tf.equal(
                                    self.labels, predictions),
                                self.mask, name='masked_equal')
                        ), tf.float32
                    ), self.total_seq_length , name='accuracy'
                )
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
    if not pathlib.Path(FLAGS.dataset).is_file():
        tf.logging.info('Generating %d sequences to %s' % \
                        (FLAGS.batch_size * FLAGS.num_batch, FLAGS.dataset))
        generator = dataset.generate_single_example(
            FLAGS.bit_width,
            FLAGS.min_seq_length,
            FLAGS.max_seq_length,
            FLAGS.num_classes)
        dataset.write_to_file(
            FLAGS.dataset, generator, FLAGS.batch_size*FLAGS.num_batch)
        tf.logging.info('Sequence generation finished')
    data = dataset.load_op(
        FLAGS.dataset, TOTAL_TIME_LENGTH, TOTAL_BIT_WIDTH, FLAGS.batch_size)
    train_op = ntm(*data, params)

    # FIXME: write summary per step only for debug purpose
    with tf.train.MonitoredTrainingSession(
            save_summaries_steps=1,
            checkpoint_dir='model') as sess:
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        # sess = tfdbg.TensorBoardDebugWrapperSession(sess, 'grpc://127.0.0.1:7000')
        for _ in range(FLAGS.train_steps):
            sess.run(train_op)

if __name__ == '__main__':
    tf.app.run()

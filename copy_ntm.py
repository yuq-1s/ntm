#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import random
from ntm import NTM

FLAGS = tf.app.flags.FLAGS

# graph = tf.Graph()
tf.app.flags.DEFINE_integer('N', 10, "memory size")
tf.app.flags.DEFINE_integer('M', 100, "memory width")
tf.app.flags.DEFINE_boolean('use_lstm', True, "usr lstm or linear controller")
tf.app.flags.DEFINE_integer('train_steps', 100, "steps to train")
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_integer('bit_width', 8, 'bits to copy per time step')
tf.app.flags.DEFINE_integer('max_seq_length', 20, 'maximum length of sequence'
                            'to copy')
tf.app.flags.DEFINE_integer('min_seq_length', 10, 'minimum length of sequence'
                            'to copy')
tf.app.flags.DEFINE_integer('random_seed', 42, 'random seed')

class CopyNTM(NTM):
    def __call__(self, inputs, labels, lengths, mode, params):
        return super().model_fn(inputs, labels, lengths, mode, params)

    @property
    def loss(self):
        if hasattr(self, '_loss'):
            return self._loss
        else:
            # with self.graph.as_default():
            self._loss = tf.losses.absolute_difference(
                self.outputs, self.labels)
            return self._loss

    @property
    def metrics(self):
        if hasattr(self, '_metrics'):
            return self._metrics
        else:
            # with self.graph.as_default():
            self._metrics = {'mae': tf.metrics.mean_absolute_error(
                labels=self.labels, predictions=self.outputs)}
            return self._metrics

def generate_single_sequence():
    ''' generate 2 continuous copy of same random bits, padded with -1 '''
    while True:
        NOT_A_WORD = -1
        seq_length = random.randint(FLAGS.min_seq_length, FLAGS.max_seq_length)
        sequence = [random.randint(0, 1) for _ in range(FLAGS.bit_width*seq_length)]
        padding_length = (FLAGS.max_seq_length-seq_length) * FLAGS.bit_width
        sequence += [NOT_A_WORD] * padding_length
        empty_sequence = [NOT_A_WORD]*FLAGS.max_seq_length*FLAGS.bit_width
        inputs = sequence + empty_sequence
        labels = empty_sequence + sequence
        yield inputs, labels, seq_length+FLAGS.max_seq_length

def get_dataset(size):
    # I swear I'll never use tfrecord again :(
    # What rubbish design and awkward interface!
    # with graph.as_default():
    data_generator = generate_single_sequence()
    data, labels, lengths = next(data_generator)
    data = tf.constant(data, dtype=tf.float32)
    data = tf.reshape(data, [FLAGS.max_seq_length*2, FLAGS.bit_width])
    labels = tf.constant(labels, dtype=tf.float32)
    labels = tf.reshape(labels, [FLAGS.max_seq_length*2, FLAGS.bit_width])
    lengths = tf.constant(lengths, dtype=tf.int64)
    return tf.train.batch([data, labels, lengths], FLAGS.batch_size)

def main(_):
    tf.logging.set_verbosity('DEBUG')
    params={
        'N': FLAGS.N,
        'M': FLAGS.M,
        'use_lstm': FLAGS.use_lstm,
        'batch_size': FLAGS.batch_size,
        'bit_width': FLAGS.bit_width}

    ntm = CopyNTM()
    train_dataset = get_dataset(FLAGS.batch_size * 10)
    eval_dataset = get_dataset(FLAGS.batch_size * 2)
    train_op = ntm(*train_dataset, tf.estimator.ModeKeys.TRAIN, params)
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir='model') as sess:
        for i in range(FLAGS.train_steps):
            sess.run(train_op)
    model = tf.estimator.Estimator(
        model_fn=CopyNTM(),
        config=tf.estimator.RunConfig(
            model_dir='model',
            tf_random_seed=FLAGS.random_seed,
        ),
        params=params)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_dataset,
        max_steps=FLAGS.train_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_dataset)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()

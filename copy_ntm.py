#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import random
from ntm import NTM

FLAGS = tf.app.flags.FLAGS

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
tf.app.flags.DEFINE_integer('num_batch', 10, 'number of batches for training')

class CopyNTM(NTM):
    def __call__(self, inputs, labels, lengths, params):
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths
        return super().model_fn(inputs, labels, lengths, params)

    @property
    def loss(self):
        if hasattr(self, '_loss'):
            return self._loss
        else:
            self._loss = tf.losses.absolute_difference(
                self.outputs, self.labels)
            tf.summary.scalar('loss', self._loss)
            return self._loss

    @property
    def metrics(self):
        if hasattr(self, '_metrics'):
            return self._metrics
        else:
            metrics = tf.metrics.mean_absolute_error(
                labels=self.labels, predictions=self.outputs)
            self._metrics = {'mae': metrics}
            tf.summary.scalar('mae', metrics)
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

def get_dataset():
    # I swear I'll never use tfrecord again :(
    # What rubbish design and awkward interface!
    # FIXME: this function is so ugly!!
    data_generator = generate_single_sequence()
    ret = {'data': [], 'labels': [], 'lengths': []}
    count = 1
    for data, labels, lengths in data_generator:
        data = np.reshape(data, [FLAGS.max_seq_length*2, FLAGS.bit_width])
        labels = np.reshape(labels, [FLAGS.max_seq_length*2, FLAGS.bit_width])
        ret['data'].append(data)
        ret['labels'].append(labels)
        ret['lengths'].append(lengths)
        count += 1
        if count > FLAGS.batch_size:
            break
    return ret['data'], ret['labels'], ret['lengths']

def main(_):
    tf.logging.set_verbosity('DEBUG')
    params={
        'N': FLAGS.N,
        'M': FLAGS.M,
        'use_lstm': FLAGS.use_lstm,
        'batch_size': FLAGS.batch_size,
        'bit_width': FLAGS.bit_width}

    # TODO: batch numbers together
    with tf.name_scope('train'):
        data_op = tf.placeholder(shape=[None, FLAGS.max_seq_length*2, FLAGS.bit_width],
                              dtype=tf.float32, name='data')
        labels_op = tf.placeholder(shape=[None, FLAGS.max_seq_length*2, FLAGS.bit_width],
                                dtype=tf.float32, name='labels')
        lengths_op = tf.placeholder(shape=[None], dtype=tf.int64, name='lengths')
        # data, labels, lengths = tf.train.batch([data, labels, lengths],
                                               # FLAGS.batch_size)
    ntm = CopyNTM()
    # train_dataset = get_dataset(FLAGS.batch_size * FLAGS.num_batch, 'train')
    # eval_dataset = get_dataset(FLAGS.batch_size * 2, 'eval')
    train_op = ntm(data_op, labels_op, lengths_op, params)
    summary_op = tf.summary.merge_all()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir='model') as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        global_step = tf.train.get_global_step()
        for _ in range(FLAGS.train_steps):
            data, labels, lengths = get_dataset()
            feed_dict = {data_op: data, labels_op: labels, lengths_op: lengths}
            summary, step, _ = sess.run([summary_op, global_step, train_op],
                                        feed_dict=feed_dict)
            writer.add_summary(summary, step)

if __name__ == '__main__':
    tf.app.run()

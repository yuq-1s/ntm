#! /usr/bin/env python3

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import numpy as np
import random
from ntm import NTM

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

def generate_single_sequence():
    def _do_generate(seq_length):
        for _i in range(seq_length):
            for _j in range(FLAGS.bit_width):
                yield random.randint(0, FLAGS.num_classes-1)
            for _j in range(2):
                yield 0

    def _pad_back(seq):
        desired = (TOTAL_BIT_WIDTH) * TOTAL_TIME_LENGTH
        seq_len_times_bit_width_plus2 = len(seq)
        assert seq_len_times_bit_width_plus2 <= desired
        return seq + [NOT_A_WORD]*(desired-seq_len_times_bit_width_plus2)

    ''' generate 2 continuous copy of same random bits, padded with NOT_A_WORD '''
    while True:
        NOT_A_WORD = FLAGS.num_classes
        seq_length = random.randint(FLAGS.min_seq_length, FLAGS.max_seq_length)
        start = [0]*FLAGS.bit_width+[0, 1]
        end = [0]*FLAGS.bit_width+[1, 0]
        sequence = list(_do_generate(seq_length))
        inputs = _pad_back(start + sequence + end)
        labels = _pad_back([NOT_A_WORD]*(seq_length+2)*(TOTAL_BIT_WIDTH) + sequence)
        lengths = seq_length*2 + 2
        yield inputs, labels, lengths

def get_dataset():
    # I swear I'll never use tfrecord again :(
    # What rubbish design and awkward interface!
    # FIXME: this function is so ugly!!
    data_generator = generate_single_sequence()
    ret = {'data': [], 'labels': [], 'lengths': []}
    count = 1
    for data, labels, lengths in data_generator:
        data = np.reshape(data, [TOTAL_TIME_LENGTH, TOTAL_BIT_WIDTH])
        labels = np.reshape(labels, [TOTAL_TIME_LENGTH, TOTAL_BIT_WIDTH])
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
        'bit_width': FLAGS.bit_width,
        'learning_rate': FLAGS.learning_rate,
        'num_classes': FLAGS.num_classes}

    # TODO: batch numbers together
    with tf.variable_scope('train'):
        data_op = tf.placeholder(shape=[None, TOTAL_TIME_LENGTH,
                                        TOTAL_BIT_WIDTH],
                              dtype=tf.float32, name='data')
        labels_op = tf.placeholder(shape=[None, TOTAL_TIME_LENGTH,
                                          TOTAL_BIT_WIDTH],
                                dtype=tf.int64, name='labels')
        lengths_op = tf.placeholder(shape=[None], dtype=tf.int64, name='lengths')
    # get_dataset()
    ntm = CopyNTM()
    train_op = ntm(data_op, labels_op, lengths_op, params)
    summary_op = tf.summary.merge_all()

    with tf.train.MonitoredTrainingSession(
            save_summaries_steps=1,
            checkpoint_dir='model') as sess:
        # sess = tfdbg.TensorBoardDebugWrapperSession(sess, 'grpc://127.0.0.1:7000')
        writer = tf.summary.FileWriter('logs', sess.graph)
        global_step = tf.train.get_global_step()
        for _ in range(FLAGS.train_steps // FLAGS.steps_per_batch):
            data, labels, lengths = get_dataset()
            for _ in range(FLAGS.steps_per_batch):
                feed_dict = {data_op: data, labels_op: labels, lengths_op: lengths}
                summary, step, _ = sess.run([summary_op, global_step, train_op],
                                            feed_dict=feed_dict)
                writer.add_summary(summary, step)

if __name__ == '__main__':
    tf.app.run()

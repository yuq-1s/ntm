#! /usr/bin/env python3

import abc
import tensorflow as tf
from ntmcell import NTMCell

# TODO: move this to config file
BATCH_SIZE = 32

# TODO: Try to let the NTM determine when to emit an output and when to halt
# TODO: Try to put the input on its memory like the original turing machine
class NTM(abc.ABC):
    def __init__(self, inputs, N, M, use_lstm, params={}):
        # inputs: [batch_size, time_step, input_dim]
        input_size = inputs.shape[2]
        # FIXME: do reshape when importing data instead of here
        inputs = tf.reshape(inputs, [BATCH_SIZE, -1, input_size])
        cell = NTMCell(input_size, N, M, use_lstm)
        initial_r = tf.get_variable('r', shape=[BATCH_SIZE, M], dtype=tf.float32)
        initial_read_w = tf.nn.softmax(tf.get_variable(
            'read_w', shape=[BATCH_SIZE, N], dtype=tf.float32))
        initial_write_w = tf.nn.softmax(tf.get_variable(
            'write_w', shape=[BATCH_SIZE, N], dtype=tf.float32))
        state = (initial_r,
                 initial_read_w,
                 initial_write_w,
                 cell.initial_read_w_controller_state,
                 cell.initial_write_w_controller_state,
                 cell.initial_erase_controller_state,
                 cell.initial_addition_controller_state)
        self.inputs = inputs
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, inputs=inputs, dtype=tf.float32, initial_state=state)
        self.model = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params=params)

    @abc.abstractmethod
    def loss(self):
        pass

    @abc.abstractmethod
    def metrics(self):
        pass

    def model_fn(self, features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(
                learning_rate=params.get('learning_rate', 1e-3)) \
                .minimize(self.loss)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=self.loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=self.loss, eval_metrics_ops=self.metrics)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions={'outputs': self.outputs})
        else:
            raise ValueError("Unknown mode %s" % mode)

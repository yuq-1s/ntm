#! /usr/bin/env python3

import abc
import tensorflow as tf
from ntmcell import NTMCell

class NTM(abc.ABC):
    def __init__(self, inputs, N, M, use_lstm, params={}):
        # inputs: [-1, time_step, input_dim]
        input_size = inputs.shape[2]
        cell = NTMCell(input_size, N, M, use_lstm)
        initial_r = tf.get_variable('r', shape=[M], dtype=tf.float32)
        initial_read_w = tf.nn.softmax(tf.get_variable(
            'read_w', shape=[N], dtype=tf.float32))
        initial_write_w = tf.nn.softmax(tf.get_variable(
            'write_w', shape=[N], dtype=tf.float32))
        initial_read_w_controller_state = tf.zeros([N+M+3]) if use_lstm else None
        initial_write_w_controller_state = tf.zeros([N+M+3]) if use_lstm else None
        initial_erase_controller_state = tf.zeros([M]) if use_lstm else None
        initial_addition_controller_state = tf.zeros([M]) if use_lstm else None
        state = (initial_r,
                 initial_read_w,
                 initial_write_w,
                 initial_read_w_controller_state,
                 initial_write_w_controller_state,
                 initial_erase_controller_state,
                 initial_addition_controller_state)
        self.inputs = inputs
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, inputs=inputs, dtype=tf.float32, initial_state=state)
        self.model = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params=params)

    @abstractmethod
    @property
    def loss(self):
        pass

    @abstractmethod
    @property
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

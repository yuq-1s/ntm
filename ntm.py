#! /usr/bin/env python3

import abc
import tensorflow as tf
from ntmcell import NTMCell

# TODO: Try to let the NTM determine when to emit an output and when to halt
# TODO: Try to put the input on its memory like the original turing machine
class NTM(abc.ABC):
    @abc.abstractmethod
    def loss(self):
        pass

    @abc.abstractmethod
    def metrics(self):
        pass

    def model_fn(self, inputs, mode, params):
        N = params['N']
        M = params['M']
        use_lstm = params['use_lstm']
        batch_size = params['batch_size']
        # inputs: [batch_size, time_step, input_dim]
        input_size = inputs.shape[2]
        # FIXME: do reshape when importing data instead of here
        inputs = tf.reshape(inputs, [batch_size, -1, input_size])
        cell = NTMCell(batch_size, input_size, N, M, use_lstm)
        initial_r = tf.get_variable('r', shape=[batch_size, M], dtype=tf.float32)
        initial_read_w = tf.nn.softmax(tf.get_variable(
            'read_w', shape=[batch_size, N], dtype=tf.float32))
        initial_write_w = tf.nn.softmax(tf.get_variable(
            'write_w', shape=[batch_size, N], dtype=tf.float32))
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
        # FIXME: Cannot use virtual method in derived classes: get this error:
        # ValueError: loss with "absolute_difference/value:0" must be from the default graph. Possible causes of this error include: 
        # 1) loss was created outside the context of the default graph.
        # 2) The object passed through to EstimatorSpec was not created in the most recent
        # call to "model_fn".
        self._loss = tf.losses.absolute_difference(self.outputs, self.labels)
        self._metrics = {'mae': tf.metrics.mean_absolute_error(
            labels=self.labels, predictions=self.outputs)}

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(
                learning_rate=params.get('learning_rate', 1e-3)) \
                .minimize(self.loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=self.loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=self.loss, eval_metric_ops=self.metrics)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions={'outputs': self.outputs})
        else:
            raise ValueError("Unknown mode %s" % mode)

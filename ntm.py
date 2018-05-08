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

    def model_fn(self, inputs, labels, lengths, params):
        N = params['N']
        M = params['M']
        use_lstm = params['use_lstm']
        batch_size = params['batch_size']
        # inputs: [batch_size, time_step, input_dim]
        input_size = inputs.shape[2]
        with tf.variable_scope('ntm'):
            with tf.variable_scope('ntm_cell'):
                cell = NTMCell(batch_size, input_size, N, M, use_lstm)
                with tf.variable_scope('states'):
                    initial_r = tf.get_variable(
                        'r', shape=[batch_size, M], dtype=tf.float32)
                    initial_read_w = tf.nn.softmax(tf.get_variable(
                        'read_w', shape=[batch_size, N], dtype=tf.float32))
                    initial_write_w = tf.nn.softmax(tf.get_variable(
                        'write_w', shape=[batch_size, N], dtype=tf.float32))
                    initial_memory = tf.get_variable(
                        'memory', shape=[batch_size, M, N], dtype=tf.float32)
                    state = (initial_r,
                            initial_read_w,
                            initial_write_w,
                            cell.initial_read_w_controller_state,
                            cell.initial_write_w_controller_state,
                            cell.initial_erase_controller_state,
                            cell.initial_addition_controller_state,
                            initial_memory)
            self.logits, self.final_state = tf.nn.dynamic_rnn(
                cell,
                sequence_length=lengths,
                inputs=inputs,
                dtype=tf.float32,
                initial_state=state)

            train_op = tf.train.AdamOptimizer(
                learning_rate=params['learning_rate']).minimize(
                    self.loss, global_step=tf.train.get_or_create_global_step())
        return train_op

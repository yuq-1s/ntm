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
        num_classes = params['num_classes']
        # inputs: [batch_size, time_step, input_dim]
        input_size = inputs.shape[2]
        with tf.variable_scope('ntm'):
            with tf.variable_scope('ntm_cell'):
                cell = NTMCell(batch_size, input_size, N, M, use_lstm,
                               num_classes)
                with tf.variable_scope('states'):
                    # FIXME: Should these states be trainable?
                    initial_r = tf.get_variable(
                        'r', shape=[batch_size, M], dtype=tf.float32,
                        initializer=tf.zeros_initializer())
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
            # To get metrics summaries
            self.metrics
        return train_op

class lstm_machine(NTM):
    def model_fn(self, inputs, labels, lengths, params):
        N = params['N']
        M = params['M']
        batch_size = params['batch_size']
        num_classes = params['num_classes']

        # create a BasicRNNCell
        cell = tf.nn.rnn_cell.BasicLSTMCell(inputs.shape[2])

        # defining initial state
        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

        # 'state' is a tensor of shape [batch_size, cell_state_size]
        # TODO: try tf.contrib.cudnn_rnn
        self.logits, self.final_state = tf.nn.dynamic_rnn(
            rnn_cell, inputs, initial_state=initial_state, dtype=tf.float32,
            sequence_length=lengths)

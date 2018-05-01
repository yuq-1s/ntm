#! /usr/bin/env python3

import tensorflow as tf

class NTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, N, M, use_lstm=True):
        self.memory = tf.get_variable("memory", shape=[N, M], dtype=tf.float32)
        self.N = N
        self.M = M
        self.head_output_size = N+M+3
        self.read_w_controller = self._new_w_controller(use_lstm,
                                                        self.head_output_size)
        self.write_w_controller = self._new_w_controller(use_lstm,
                                                         self.head_output_size)
        self.erase_controller = self._new_w_controller(use_lstm, M)
        self.addition_controller = self._new_w_controller(use_lstm, M)
        self.encode = tf.get_variable('encode', shape=[input_size, 2*M])
        self.decode = tf.get_variable('decode', shape=[2*M, input_size])

    @property
    def state_size(self):
        # FIXME: What is this for?
        return 43

    @property
    def output_size(self):
        # FIXME: What is this for?
        return 42

    def _get_w(self, last_w, raw_output):
        k, beta, g, s, gamma = tf.split(raw_output,
                                        [self.M, 1, 1, self.N, 1],
                                        axis=1)
        memory_row_norm = tf.reduce_sum(tf.abs(self.memory), axis=1)
        logits = beta * tf.matmul(self.memory, k) / memory_row_norm
        w_c = tf.nn.softmax(logits)
        w_g = g * w_c + (1-g) * last_w
        w_g = tf.cast(w_g, tf.complex64)
        s = tf.cast(s, tf.complex64)
        w_tild = tf.real(tf.ifft(tf.fft(w_g), tf.fft(s)))
        w = tf.nn.softmax(gamma * tf.log(w_tild))
        return w

    def _new_w_controller(self, use_lstm, output_size):
        if use_lstm:
            return tf.contrib.rnn.BasicLSTMCell(output_size)
        else:
            def linear_controller(inputs, dummy_state):
                assert dummy_state is None
                output = tf.layers.dense(inputs, output_size)
                return output, dummy_state
            return linear_controller

    def read_head(self, inputs, last_w, read_w_controller_state):
        # inputs: of shape [-1, 2, M] is embedding of input and last_r
        with tf.variable_scope("read_head"):
            raw_output, controller_state = self.read_w_controller(
                inputs, controller_state)
            w = self._get_w(last_w, raw_output)
            r = w * self.memory
            return r, w, controller_state

    def write_head(self, inputs, last_w, write_w_controller_state,
                   erase_controller_state, addition_controller_state):
        def outer_product(a, b):
            return tf.reshape(a, [-1, 1]) * tf.reshape(b, [1, -1])

        with tf.variable_scope("write_head"):
            raw_output, write_w_controller_state = self.write_w_controller(
                inputs, write_w_controller_state)
            e, erase_controller_state = self.erase_controller(
                inputs, write_w_controller_state)
            # squash e to (0, 1)
            e = tf.sigmoid(e)
            a, addition_controller_state = self.addition_controller(
                inputs, write_w_controller_state)
            w = self._get_w(last_w, raw_output)
            memory_tild = self.memory - self.memory * outer_product(w, e)
            write_op = self.memory.assign(memory_tild + outer_product(w, a))
            return (write_op, w, write_w_controller_state,
                    erase_controller_state, addition_controller_state)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            (last_r,
             last_read_w,
             last_write_w,
             read_w_controller_state,
             write_w_controller_state,
             last_erase_state,
             last_addition_state) = state
            input_embedding = tf.matmul(inputs, self.encode)
            inputs_vector = tf.matmul(input_embedding, inputs)
            inputs_vector = tf.stack(inputs_vector, last_r)
            write_op, write_w, write_w_controller_state, erase_controller_state,
            addition_controller_state = self.write_head(
                inputs_vector, last_write_w, write_w_controller_state,
                erase_controller_state, addition_controller_state)
            # To make sure write_op is executed
            with tf.control_dependencies([write_op]):
                r, read_w, read_w_controller_state = self.read_head(
                    inputs_vector, last_read_w, read_w_controller_state)

            return tf.matmul(self.decode, inputs_vector),
        (r, read_w, write_w, read_w_controller_state,
         write_w_controller_state, erase_controller_state,
         addition_controller_state)

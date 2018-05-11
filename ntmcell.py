#! /usr/bin/env python3

import tensorflow as tf

class NTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, input_dim, N, M, use_lstm=True, num_classes=2):
        self.N = N
        self.M = M
        self.head_output_size = N+M+3
        self.num_classes = num_classes+1
        self.batch_size = batch_size
        self.read_w_controller = self._new_w_controller(use_lstm,
                                                        self.head_output_size)
        self.write_w_controller = self._new_w_controller(use_lstm,
                                                         self.head_output_size)
        self.erase_controller = self._new_w_controller(use_lstm, M)
        self.addition_controller = self._new_w_controller(use_lstm, M)
        self.input_dim = int(input_dim)
        # self.encoder = tf.get_variable('encoder', shape=[2, 1])
        self.decoder = tf.get_variable('decoder', shape=[M, input_dim *
                                                         self.num_classes])
        if use_lstm:
            self.initial_read_w_controller_state = \
                self._get_lstm_initial_state(self.head_output_size, 'read_w')
            self.initial_write_w_controller_state = \
                self._get_lstm_initial_state(self.head_output_size, 'write_w')
            self.initial_erase_controller_state = \
                self._get_lstm_initial_state(M, 'erase')
            self.initial_addition_controller_state = \
                self._get_lstm_initial_state(M, 'addition')
        else:
            # dummy states
            self.initial_read_w_controller_state = tf.constant([])
            self.initial_write_w_controller_state = tf.constant([])
            self.initial_erase_controller_state = tf.constant([])
            self.initial_addition_controller_state = tf.constant([])

    def _get_lstm_initial_state(self, hidden_size, scope):
        with tf.variable_scope(scope):
            return tf.contrib.rnn.LSTMStateTuple(
                tf.get_variable(
                    name=scope+'_h', shape=[self.batch_size, hidden_size]),
                tf.get_variable(
                    name=scope+'_c', shape=[self.batch_size, hidden_size])
            )

    @property
    def state_size(self):
        # FIXME: What is this for?
        return 43

    @property
    def output_size(self):
        return self.input_dim * self.num_classes

    def _get_w(self, last_w, raw_output, memory):
        with tf.variable_scope('get_w'):
            k, beta, g, s, gamma = tf.split(raw_output,
                                            [self.M, 1, 1, self.N, 1],
                                            axis=1)
            memory_row_norm = tf.reduce_sum(tf.abs(memory), axis=1)
            foo = tf.squeeze(tf.matmul(tf.expand_dims(k, 1), memory))
            logits = beta * foo / memory_row_norm
            w_c = tf.nn.softmax(logits)
            w_g = g * w_c + (1-g) * last_w
            w_g = tf.cast(w_g, tf.complex64)
            s = tf.cast(s, tf.complex64)
            w_tild = tf.real(tf.ifft(tf.fft(w_g) * tf.fft(s)))
            # FIXME: tf.log yields lots of NaN here!!
            # w = tf.nn.softmax(gamma * tf.log(w_tild))
            w = tf.nn.softmax(gamma * w_tild)
            return w

    def _new_w_controller(self, use_lstm, output_size):
        if use_lstm:
            return tf.contrib.rnn.BasicLSTMCell(output_size)
        else:
            def linear_controller(inputs, dummy_state):
                output = tf.layers.dense(inputs, output_size)
                return output, dummy_state
            return linear_controller

    def read_head(self, inputs, last_w, controller_state, memory):
        # inputs: of shape [-1, M+self.input_dim] is the concatenation of inputs
        # and last_r
        with tf.variable_scope("read_head"):
            raw_output, controller_state = self.read_w_controller(
                inputs, controller_state)
            w = self._get_w(last_w, raw_output, memory)
            r = tf.squeeze(tf.matmul(memory, tf.expand_dims(w, 2)))
            return r, w, controller_state

    def write_head(self, inputs, last_w, write_w_controller_state,
                   erase_controller_state, addition_controller_state, memory):
        def outer_product(a, b):
            return tf.reshape(a, [self.batch_size, -1, 1]) * \
                tf.reshape(b, [self.batch_size, 1, -1])

        with tf.variable_scope("write_head"):
            raw_output, write_w_controller_state = self.write_w_controller(
                inputs, write_w_controller_state)
            w = self._get_w(last_w, raw_output, memory)
            with tf.variable_scope('erase'):
                e, erase_controller_state = self.erase_controller(
                    inputs, erase_controller_state)
                # squash e to (0, 1)
                e = tf.sigmoid(e, name='squash')
            with tf.variable_scope('addition'):
                a, addition_controller_state = self.addition_controller(
                    inputs, addition_controller_state)
                memory_tild = memory - memory * outer_product(e, w)
                new_memory = memory_tild + outer_product(a, w)
            return (w, write_w_controller_state,
                    erase_controller_state, addition_controller_state,
                    new_memory)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            (last_r,
             last_read_w,
             last_write_w,
             read_w_controller_state,
             write_w_controller_state,
             last_erase_state,
             last_addition_state,
             memory) = state
            # FIXME: how to encode inputs??
            inputs = tf.concat([inputs, last_r], axis=1)
            # inputs = tf.concat(
                # [tf.nn.embedding_lookup(self.encoder, inputs), last_r], axis=1)
            # inputs = tf.matmul(inputs, self.encoder) + last_r

            # READ
            r, read_w, read_w_controller_state = self.read_head(
                inputs, last_read_w, read_w_controller_state, memory)

            # WRITE
            (write_w,
             write_w_controller_state,
             erase_controller_state,
             addition_controller_state,
             memory) = self.write_head(
                 inputs, last_write_w, write_w_controller_state,
                 last_erase_state, last_addition_state, memory)

            logits = tf.matmul(r, self.decoder, name='logits')
            return logits, (r, read_w, write_w, read_w_controller_state,
                             write_w_controller_state, erase_controller_state,
                             addition_controller_state, memory)

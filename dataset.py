import numpy as np
import tensorflow as tf
import random
import pathlib
import itertools

def generate_single_example(
        bit_width,
        min_seq_length,
        max_seq_length,
        num_classes):
    total_bit_width = bit_width+2
    total_time_length = max_seq_length*2+2
    NOT_A_WORD = num_classes
    def _do_generate(seq_length):
        for _i in range(seq_length):
            for _j in range(bit_width):
                yield random.randint(0, num_classes-1)
            for _j in range(2):
                yield 0

    def _pad_back(seq):
        desired = (total_bit_width) * total_time_length
        seq_len_times_bit_width_plus2 = len(seq)
        assert seq_len_times_bit_width_plus2 <= desired
        return seq + [NOT_A_WORD]*(desired-seq_len_times_bit_width_plus2)

    def _generate_single_record():
        ''' generate 2 continuous copy of same random bits, padded with NOT_A_WORD '''
        while True:
            NOT_A_WORD = num_classes
            seq_length = random.randint(min_seq_length, max_seq_length)
            start = [0]*bit_width+[0, 1]
            end = [0]*bit_width+[1, 0]
            sequence = list(_do_generate(seq_length))
            inputs = _pad_back(start + sequence + end)
            labels = _pad_back([NOT_A_WORD]*(seq_length+2)*(total_bit_width) + sequence)
            lengths = seq_length*2 + 2
            yield inputs, labels, lengths

    def _float_list(a):
        # a: list
        return tf.train.Feature(float_list=tf.train.FloatList(value=a))

    def _int64_feature(a):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=a))

    generator = _generate_single_record()
    for (inputs, labels, lengths) in generator:
        features = tf.train.Features(feature={
            'inputs': _float_list(inputs),
            'labels': _int64_feature(labels),
            'lengths': _int64_feature([lengths])})
        yield tf.train.Example(features=features)

def write_to_file(filename, generator, dataset_size):
    pathlib.Path(filename).parents[0].mkdir(parents=True, exist_ok=True)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for example in itertools.islice(generator, dataset_size):
            writer.write(example.SerializeToString())

def load_op(filename, total_time_length, total_bit_width, batch_size):
    def process_record(record):
        with tf.name_scope('process_record'):
            feature = {'inputs': tf.FixedLenFeature([total_time_length*total_bit_width], tf.float32),
                    'labels': tf.FixedLenFeature([total_time_length*total_bit_width], tf.int64),
                    'lengths': tf.FixedLenFeature([1], tf.int64)}
            features = tf.parse_single_example(record, features=feature)
            inputs = tf.reshape(
                features['inputs'], [total_time_length, total_bit_width])
            labels = tf.reshape(
                features['labels'], [total_time_length, total_bit_width])
            lengths = tf.reshape(features['lengths'], [])
            return inputs, labels, lengths

    return tf.data.TFRecordDataset(filename) \
        .shuffle(256) \
        .repeat() \
        .map(process_record) \
        .batch(batch_size) \
        .make_one_shot_iterator().get_next()

if __name__ == '__main__':
    filename = 'data/train.tfrecord'
    generator = generate_single_example(8, 10, 20, 2)
    # write_to_file(filename, generator, 32)
    inputs, labels, lengths = tf.data.TFRecordDataset(filename) \
        .map(process_record) \
        .batch(4) \
        .repeat().make_one_shot_iterator().get_next()

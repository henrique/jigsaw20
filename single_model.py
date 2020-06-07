""" build a TFAutoModel and load its data from npz or tfrec dataset """
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model

from transformers import TFAutoModel

from GeM import GeneralizedMeanPooling1D


def build_model(model_id='jplu/tf-xlm-roberta-large', from_pt=False, transformer=None,
                max_len=192, dropout=0.2, pooling='first',
                **_):
    """ build a TFAutoModel """
    if transformer is None:
        transformer = TFAutoModel.from_pretrained(model_id, from_pt=from_pt)

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]

    if pooling == 'first':
        cls_token = sequence_output[:, 0, :]
    elif pooling == 'max':
        cls_token = GlobalMaxPooling1D()(sequence_output)
    elif pooling == 'avg':
        cls_token = GlobalAveragePooling1D()(sequence_output)
    elif pooling == 'GeM':
        cls_token = GeneralizedMeanPooling1D(p=3)(sequence_output)

    if dropout > 0:
        cls_token = Dropout(dropout)(cls_token)

    out = Dense(1, activation='sigmoid')(cls_token)
    model = Model(inputs=input_word_ids, outputs=out)

    return model



def np_dataset(dataset, batch_size, seed):
    """ load npz datasets """
    array = np.load(dataset)
    x_train, x_valid, x_test, y_train, y_valid = [array[k] for k in list(array)]
    # Shuffle
    x_train = pd.DataFrame(np.concatenate([x_train.T, [y_train]]).T
                          ).sample(frac=1, random_state=seed).values
    assert abs(x_train[..., :-1] - x_train[..., :-1].astype('int32')).max() == 0
    x_train, y_train = x_train[..., :-1].astype('int32'), x_train[..., -1].astype('float32')
    print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape)

    ## Set Datasets
    auto_tune = tf.data.experimental.AUTOTUNE
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(batch_size)
        .cache()
        .prefetch(auto_tune)
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    return train_dataset, valid_dataset, test_dataset


def val_np_dataset(dataset='../input/jigsaw20-val-test-ds/jigsaw20_val_ds.npz', batch_size=128):
    """ load npz datasets """
    array = np.load(dataset)
    x_valid, x_test, y_valid = [array[k] for k in list(array)]
    print(x_valid.shape, x_test.shape, y_valid.shape)

    ## Set Datasets
    auto_tune = tf.data.experimental.AUTOTUNE
    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(batch_size)
        .cache()
        .prefetch(auto_tune)
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    return valid_dataset, test_dataset


def tf_dataset(dataset, batch_size, max_len, seed):
    """ load tfrec datasets """
    auto_tune = tf.data.experimental.AUTOTUNE

    train_dataset = (
        load_tf_dataset(dataset+'train*.tfrec', max_len, seed)
        .repeat()
        .shuffle(2048)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    valid_dataset, test_dataset = val_np_dataset(batch_size=batch_size)

    return train_dataset, valid_dataset, test_dataset

def load_tf_dataset(filenames, max_len, seed, ordered=False):
    """ load a tfrec dataset """
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    auto_tune = tf.data.experimental.AUTOTUNE

    def read_labeled_tfrecord(example, max_len=max_len):
        """ decode a tfrec """
        tf_format = {
            "data": tf.io.FixedLenFeature(max_len, tf.int64),
            "label": tf.io.FixedLenFeature([], tf.float32),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, tf_format)
        return example['data'], example['label']

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    # expand and shuffle files
    filenames = tf.io.gfile.glob(filenames)
    random.Random(seed).shuffle(filenames)
    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=auto_tune)
    # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=auto_tune)
    return dataset

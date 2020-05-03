""" build a dual TFAutoModel and load its data from npz or tfrec dataset """
import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

from transformers import TFAutoModel


def build_model(model_id1='bert-base-multilingual-cased',
                model_id2='bert-base-multilingual-uncased',
                max_len=192, dropout=0.2,
                **_):
    """ build a dual TFAutoModel """
    print(model_id1, model_id2)

    transformer1 = TFAutoModel.from_pretrained(model_id1)
    transformer2 = TFAutoModel.from_pretrained(model_id2)

    input_word_ids1 = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids1")
    out1 = transformer1(input_word_ids1)

    input_word_ids2 = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids2")
    out2 = transformer2(input_word_ids2)

    sequence_output1 = out1[0]
    sequence_output2 = out2[0]
    cls_token1 = sequence_output1[:, 0, :]
    cls_token2 = sequence_output2[:, 0, :]

    x = Dropout(dropout)(cls_token1) + Dropout(dropout)(cls_token2)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_word_ids1, input_word_ids2], outputs=out)

    return model



def np_dataset(datasets, batch_size, seed):
    """ load npz datasets """
    datasets = sorted(glob.glob(datasets))
    print(datasets)

    ## x_train1
    array = np.load(datasets[0])
    x_train, x_valid, x_test, y_train, y_valid = [array[k] for k in list(array)]
    # Shuffle
    x_train = pd.DataFrame(np.concatenate([x_train.T, [y_train]]).T
                          ).sample(frac=1, random_state=seed).values
    assert abs(x_train[..., :-1] - x_train[..., :-1].astype('int32')).max() == 0
    x_train, y_train = x_train[..., :-1].astype('int32'), x_train[..., -1].astype('float32')
    print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape)

    x_train1, x_valid1, x_test1 = x_train, x_valid, x_test

    ## x_train2
    array = np.load(datasets[1])
    x_train, x_valid, x_test, y_train, y_valid = [array[k] for k in list(array)]
    # Shuffle
    x_train = pd.DataFrame(np.concatenate([x_train.T, [y_train]]).T
                          ).sample(frac=1, random_state=seed).values
    assert abs(x_train[..., :-1] - x_train[..., :-1].astype('int32')).max() == 0
    x_train, y_train = x_train[..., :-1].astype('int32'), x_train[..., -1].astype('float32')
    print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape)

    x_train2, x_valid2, x_test2 = x_train, x_valid, x_test

    ## Set Datasets
    auto_tune = tf.data.experimental.AUTOTUNE
    train_dataset = (
        tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((x_train1, x_train2)),
            tf.data.Dataset.from_tensor_slices(y_train)
        ))
        .repeat()
        .shuffle(2048)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    valid_dataset = (
        tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((x_valid1, x_valid2)),
            tf.data.Dataset.from_tensor_slices(y_valid)
        ))
        .batch(batch_size)
        .cache()
        .prefetch(auto_tune)
    )

    test_dataset = (
        tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((x_test1, x_test2)),
            tf.data.Dataset.from_tensor_slices(np.ones(len(x_test1), dtype=np.float32))
        ))
        .batch(batch_size)
        .prefetch(auto_tune)
    )

    return train_dataset, valid_dataset, test_dataset


def tf_dataset(*_):
    """ load tfrec datasets """
    raise "Not Implemented"

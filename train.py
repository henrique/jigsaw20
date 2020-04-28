""" build and train a TFAutoModel from npz or tfrec dataset """

import os
import gc
import time
import random

import logging
import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from google.cloud import storage

import tensorflow_addons as tfa
# from tensorflow_addons.optimizers.utils import fit_bn

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

from transformers import TFAutoModel
from one_cycle_scheduler import OneCycleScheduler

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def focal_loss(gamma=2., pos_weight=1, label_smoothing=0.05):
    """ binary focal loss with label_smoothing """
    def binary_focal_loss(labels, p):
        """ bfl clojure """
        labels = tf.dtypes.cast(labels, dtype=p.dtype)
        if label_smoothing is not None:
            labels = (1 - label_smoothing) * labels + label_smoothing * 0.5

        # Predicted probabilities for the negative class
        q = 1 - p

        # For numerical stability (so we don't inadvertently take the log of 0)
        p = tf.math.maximum(p, K.epsilon())
        q = tf.math.maximum(q, K.epsilon())

        # Loss for the positive examples
        pos_loss = -(q ** gamma) * tf.math.log(p) * pos_weight

        # Loss for the negative examples
        neg_loss = -(p ** gamma) * tf.math.log(q)

        # Combine loss terms
        loss = labels * pos_loss + (1 - labels) * neg_loss

        return loss

    return binary_focal_loss


def build_model(model_id='jplu/tf-xlm-roberta-large', max_len=192,
                optimizer='LAMB', lr=2e-5, weight_decay=1e-6,
                loss_fn='bce', label_smoothing=0.01,
                pos_weight=5, gamma=2.0,  ## focal loss
                dropout=0.2, amp=False,
                **_):
    """ build a TFAutoModel """
    transformer = TFAutoModel.from_pretrained(model_id)

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    if dropout > 0:
        cls_token = Dropout(dropout)(cls_token)
    out = Dense(1, activation='sigmoid')(cls_token)
    model = Model(inputs=input_word_ids, outputs=out)

    if loss_fn == 'focal':
        loss = focal_loss(pos_weight=pos_weight, gamma=gamma, label_smoothing=label_smoothing)
    elif loss_fn == 'bce':
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

    if optimizer == 'LAMB':
        opt = tfa.optimizers.LAMB(lr=lr, weight_decay_rate=weight_decay)
    elif optimizer == 'AdamW':
        opt = tfa.optimizers.AdamW(lr=lr, weight_decay=weight_decay)

    if amp:
        print('Using auto_mixed_precision.')
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                ]
    )

    return model


def save_fig(filename, path, gcs):
    """ save current plt fig to gcs """
    plt.gcf().savefig(filename)
    plt.close()
    # init GCS client and upload file
    client = storage.Client()
    bucket = client.get_bucket(gcs)
    blob = bucket.blob(f'{path}/{filename}')
    blob.upload_from_filename(filename=filename)


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

    valid_dataset = (
        load_tf_dataset(dataset+'valid*.tfrec', max_len, seed)
        .batch(batch_size)
        .cache()
        .prefetch(auto_tune)
    )

    test_dataset = (
        load_tf_dataset(dataset+'test*.tfrec', max_len, seed)
        .batch(batch_size)
        .prefetch(auto_tune)
    )

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


def train_model(model, strategy, checkpoint_path,
                dataset, max_len=192, seed=0,
                epochs=30, steps_per_epoch=250,
                lr=2e-5, one_cycle=True, warm_up=1,
                mom_min=0.85, mom_max=0.95,
                div_factor=100, final_div_factor=250,
                batch_size=28, callback=None,
                **_):
    """ train the given model """
    batch_size = batch_size * strategy.num_replicas_in_sync
    print('batch_size:', batch_size)

    if dataset.startswith('gs://'):
        train_dataset, valid_dataset, test_dataset = tf_dataset(dataset, batch_size, max_len, seed)
    else:
        train_dataset, valid_dataset, test_dataset = np_dataset(dataset, batch_size, seed)

    ## Train
    callbacks = [] if callback is None else [callback]
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=1e-4,
                                                      mode='max', patience=6, verbose=1,
                                                      restore_best_weights=True))

    if one_cycle:
        callbacks.append(OneCycleScheduler(lr_max=lr, steps=steps_per_epoch*epochs,
                                           mom_min=mom_min, mom_max=mom_max,
                                           phase_1_pct=warm_up/epochs,
                                           div_factor=div_factor,
                                           final_div_factor=final_div_factor))
    else:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.31,
                                                              patience=2, cooldown=1, mode='max',
                                                              verbose=1, min_delta=1e-4))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_auc',
                                                        verbose=1, mode='max',
                                                        save_best_only=True,
                                                        save_weights_only=True))
    print(callbacks)

    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    # load best
    if epochs > 1:
        # latest = tf.train.latest_checkpoint(checkpoint_dir)
        with strategy.scope():
            model.load_weights(checkpoint_path)

    return (model,
            model.predict(valid_dataset, verbose=1),
            model.predict(test_dataset, verbose=1))


def plot_history(history, path, bucket):
    """ plot a save the model's history """
    ## Eval
    _, axs = plt.subplots(1, 3, figsize=(18, 4))
    # Plot training & validation loss values
    ax = axs[0]
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='lower left')

    # Plot training & validation accuracy values
    ax = axs[1]
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation accuracy values
    ax = axs[2]
    ax.plot(history.history['auc'])
    ax.plot(history.history['val_auc'])
    ax.set_title('Model AUC')
    ax.set_ylabel('AUC')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')

    save_fig('history.png', path, bucket)


def setup_tpu(tpu_id):
    """ resolve a tpu cluster """
    if tpu_id is None:
        with open('tpu', 'r') as content_file:
            tpu_id = content_file.read()
            print(dict(tpu_id=tpu_id))

    ## Detect hardware, return appropriate distribution strategy
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_id)
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy



##################
###### MAIN ######
##################

def train(gcs='hm-eu-w4', path='jigsaw/test',
          seed=0, max_len=192, tpu_id=None,
          **kwargs):
    """ build and train a TFAutoModel from npz or tfrec dataset """
    params = dict(locals())
    params.update(kwargs)
    params = pd.DataFrame(params, index=[0])
    kw_params = params.T[0].to_dict()
    print(params.T)
    gc.collect()

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    strategy = setup_tpu(tpu_id)

    ## Configuration
    path = f'{path}/{time.strftime("%Y%m%d_%H%M%S")}_{tpu_id}'
    gcs_path = f'gs://{gcs}/{path}'
    checkpoint_path = f"{gcs_path}/best_model.tf"
    print('gcs_path:', gcs_path)

    ## Load and Train
    with strategy.scope():
        model = build_model(**kw_params)
    model, preds, sub = train_model(model, strategy, checkpoint_path, **kw_params)

    ## Save results
    plot_history(model.history, path, gcs)
    history = pd.DataFrame(model.history.history)
    print(history)
    history.to_csv(f'{gcs_path}/history.csv', index=False)

    ## Load Dataset
    comp_ds = '../input/jigsaw-multilingual-toxic-comment-classification'
    valid = pd.read_csv(f'{comp_ds}/validation.csv')
#     test = pd.read_csv(f'{comp_ds}/test.csv')
    sub = pd.read_csv(f'{comp_ds}/sample_submission.csv')

    valid['pred'] = preds
    valid.to_csv(f'{gcs_path}/valid_oof.csv', index=False)

    valid.groupby('toxic').pred.hist(bins=100, log=True, alpha=0.5)
    plt.legend([0, 1])
    save_fig('valid_hist.png', path, gcs)

    valid[valid.toxic == 1].groupby('lang').pred.hist(bins=50, log=True, alpha=0.34)
    plt.legend(valid.lang.unique())
    save_fig('valid_toxic_hist.png', path, gcs)

    valid_auc = roc_auc_score(valid.toxic, valid.pred)
    print('AUC:', valid_auc,
          'toxic:', valid.toxic.mean(),
          'pred:', valid.pred.mean(),
          'ratio:', (valid.pred > 0.5).mean())

    # over sample toxic
    bal_valid = valid.append(valid[valid.toxic == 1], ignore_index=True)
    print('AUC_bal:', roc_auc_score(bal_valid.toxic, bal_valid.pred),
          'toxic:', bal_valid.toxic.mean(),
          'pred:', bal_valid.pred.mean(),
          'ratio:', (bal_valid.pred > 0.5).mean())

    ## Submission
    sub['toxic'] = sub
    sub.to_csv(f'{gcs_path}/submission.csv', index=False)

    sub.toxic.hist(bins=100, log=True)
    save_fig('sub_hist.png', path, gcs)
    print('mean:', sub.toxic.mean(), 'ratio:', (sub.toxic > 0.5).mean())

    ## Save params
    params['auc'] = valid_auc
    params.to_csv(f'{gcs_path}/params{valid_auc:04f}.csv', index=False)
    print(params.T)

    return valid_auc

# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import argparse
import codecs
import glob
import json
import logging
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
# from tensorflow_examples.models.pix2pix import pix2pix

# try:
#     import tensorflow_addons as tfa
# except ImportError:
#     subprocess.check_call([sys.executable, "-m", "pip",
#                            "install", 'tensorflow_addons'])
# finally:
#     import tensorflow_addons as tfa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        interval = int(os.environ['SAVE_INTERVAL'])
        if epoch % interval == 0:
            _show_predictions(epoch)
            logging.info(
                '\nSample Prediction after epoch {}\n'.format(epoch+interval))


def save_history(path, history):

    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            history_for_json[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
            if type(history.history[key][0]) == np.float32 or type(history.history[key][0]) == np.float64:
                history_for_json[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(
            ',', ':'), sort_keys=True, indent=4)


def save_model(model, output):

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
#     model.save(output, 'last_model.h5')
    tf.keras.models.save_model(model, output)
    logging.info("Model successfully saved at: {}".format(output))
    return


def model(args):

    # -- Keras Functional API -- #
    # -- UNet Implementation -- #
    # Everything here is from tensorflow.keras.layers
    # I imported tensorflow.keras.layers * to make it easier to read
    dropout_rate = 0.5
    input_size = (args.RESIZE_WIDTH, args.RESIZE_HEIGHT, 3)

    # If you want to know more about why we are using `he_normal`:
    # https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849#319849
    # Or the excellent fastai course:
    # https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb
    initializer = 'he_normal'

    # Block encoder 1
    inputs = Input(shape=input_size)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(inputs)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_enc_1)

    # Block encoder 2
    max_pool_enc_2 = AveragePooling2D(pool_size=(2, 2))(conv_enc_1)
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(max_pool_enc_2)
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_enc_2)

    # Block  encoder 3
    max_pool_enc_3 = AveragePooling2D(pool_size=(2, 2))(conv_enc_2)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(max_pool_enc_3)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_enc_3)

    # Block  encoder 4
    max_pool_enc_4 = AveragePooling2D(pool_size=(2, 2))(conv_enc_3)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(max_pool_enc_4)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_enc_4)
    # -- Encoder -- #

    # ----------- #
    maxpool = AveragePooling2D(pool_size=(2, 2))(conv_enc_4)
    conv = Conv2D(1024, 3, activation='relu', padding='same',
                  kernel_initializer=initializer)(maxpool)
    conv = Conv2D(1024, 3, activation='relu', padding='same',
                  kernel_initializer=initializer)(conv)
    # ----------- #

    # -- Decoder -- #
    # Block decoder 1
    up_dec_1 = Conv2D(512, 2, activation='relu', padding='same',
                      kernel_initializer=initializer)(UpSampling2D(size=(2, 2))(conv))
    merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis=3)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(merge_dec_1)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_dec_1)

    # Block decoder 2
    up_dec_2 = Conv2D(256, 2, activation='relu', padding='same',
                      kernel_initializer=initializer)(UpSampling2D(size=(2, 2))(conv_dec_1))
    merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis=3)
    conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(merge_dec_2)
    conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_dec_2)

    # Block decoder 3
    up_dec_3 = Conv2D(128, 2, activation='relu', padding='same',
                      kernel_initializer=initializer)(UpSampling2D(size=(2, 2))(conv_dec_2))
    merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis=3)
    conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(merge_dec_3)
    conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_dec_3)

    # Block decoder 4
    up_dec_4 = Conv2D(64, 2, activation='relu', padding='same',
                      kernel_initializer=initializer)(UpSampling2D(size=(2, 2))(conv_dec_3))
    merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis=3)
    conv_dec_4 = Conv2D(64, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(merge_dec_4)
    conv_dec_4 = Conv2D(64, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_dec_4)
    conv_dec_4 = Conv2D(2, 3, activation='relu', padding='same',
                        kernel_initializer=initializer)(conv_dec_4)
    # -- Dencoder -- #

    output = Conv2D(args.OUTPUT_CHANNELS, 1, activation='softmax')(conv_dec_4)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    # here I'm using a new optimizer: https://arxiv.org/abs/1908.03265
    # optimizer = tfa.optimizers.RectifiedAdam(
    #     lr=1e-3,
    #     total_steps=args.total_step,
    #     warmup_proportion=0.1,
    #     min_lr=1e-5,
    # )
#     lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(
#         3e-4, args.total_step//3)
#     print("lr_fn : {}".format(lr_fn))
#     print("args.total_step : {}".format(args.total_step))
    lr_fn = 3e-4
    optimizer = Adam(learning_rate=lr_fn)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def get_filenames(args, channel_name):
    if channel_name in ['train', 'test']:
        return [file_names for file_names in glob.glob(args.data_dir+'/*') if args.DATASET_NAME + '-' + channel_name + '.tfrecord' in file_names]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)


def train_input_fn(args):
    return _input(args, 'train')


def test_input_fn(args):
    return _input(args, 'test')


def _input(args, channel_name):
    try:
        mode_channel_name = channel_name+'ing' if channel_name == 'train' else channel_name
        mode = args.data_config[mode_channel_name]['TrainingInputMode']
    except:
        mode = 'File'

    """Uses the tf.data input pipeline for dataset.
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    filenames = get_filenames(args, channel_name)
    # Repeat infinitely.
    logging.info("Running {} in {} mode".format(channel_name, mode))
    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(
            channel=channel_name, record_format='TFRecord')
    else:
        dataset = tf.data.TFRecordDataset(filenames)

    # Potentially shuffle records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.map(
            _load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        buffer_size = int(args.train_num_examples * 0.4) + 3 * args.BATCH_SIZE

        dataset = dataset.cache().shuffle(
            buffer_size=buffer_size).batch(args.BATCH_SIZE).repeat()

    elif channel_name == 'test':
        dataset = dataset.map(_load_image_test)

        for image, mask in dataset.take(1):
            sample_image, sample_mask = image, mask
        
        _img_save('sample_image.jpg', sample_image)
        _img_save('sample_mask.png', sample_mask)

        dataset = dataset.batch(args.BATCH_SIZE)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def _normalize(input_image, input_mask):
    input_image = input_image / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def _load_image_train(data):
    input_image, input_mask = _parse_function(data)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = _normalize(input_image, input_mask)

    return input_image, input_mask


def _load_image_test(data):
    input_image, input_mask = _parse_function(data)
    input_image, input_mask = _normalize(input_image, input_mask)

    return input_image, input_mask


def _parse_function(data):
    """Parse record from value."""
    featdef = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'segmentation_mask': tf.io.FixedLenFeature([], tf.string)
    }

    datapoint = tf.io.parse_single_example(data, featdef)

    input_image = tf.cast(tf.io.decode_image(
        datapoint['image'], channels=3, expand_animations=False), tf.float32)
    input_mask = tf.cast(tf.io.decode_image(
        datapoint['segmentation_mask'], channels=1, expand_animations=False), tf.float32)

    input_image = tf.image.resize(
        input_image, (args.RESIZE_WIDTH, args.RESIZE_HEIGHT))
    input_mask = tf.image.resize(
        input_mask, (args.RESIZE_WIDTH, args.RESIZE_HEIGHT))

    return input_image, input_mask


def _create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    print("tf.newaxis : {}".format(tf.newaxis))
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def _img_save(img_name, image, seq=None):
    if seq is not None:
        path = os.path.join(
            os.environ['output_data_dir'], str(seq) + "-" + img_name)
    else:
        path = os.path.join(os.environ['output_data_dir'], img_name)
    tf.keras.preprocessing.image.save_img(path, image)


def _img_open(img_name):
    path = os.path.join(os.environ['output_data_dir'], img_name)
    return tf.keras.preprocessing.image.img_to_array(Image.open(path))


def _show_predictions(epoch):
    sample_image = _img_open('sample_image.jpg')
    predicted_mask = _create_mask(
        model.predict(sample_image[tf.newaxis, ...]))

    predicted_mask = tf.image.resize(predicted_mask, (128, 128))
    _img_save('predicted_mask.png', predicted_mask, seq=epoch)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--OUTPUT_CHANNELS', type=int, default=3)
    parser.add_argument('--EPOCHS', type=int, default=20)
    parser.add_argument('--VAL_SUBSPLITS', type=int, default=1)

    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--BUFFER_SIZE', type=int, default=1000)
    parser.add_argument('--DATASET_NAME', type=str)
    parser.add_argument('--SAVE_INTERVAL', type=int, default=1)
    parser.add_argument('--RESIZE_WIDTH', type=int, default=128)
    parser.add_argument('--RESIZE_HEIGHT', type=int, default=128)
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--artifact_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str,
                        default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--data-config', type=json.loads,
                        default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--output_data_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--fw-params', type=json.loads,
                        default=os.environ.get('SM_FRAMEWORK_PARAMS'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = parse_args()

    mpi = False

    args.artifact_dir = os.path.join(args.artifact_dir, '000000001')
    os.environ['output_data_dir'] = args.output_data_dir
    os.environ['SAVE_INTERVAL'] = str(args.SAVE_INTERVAL)

    args.SAVE_FLAG = False

    if 'sagemaker_mpi_enabled' in args.fw_params:
        if args.fw_params['sagemaker_mpi_enabled']:
            import horovod.tensorflow.keras as hvd
            mpi = True
            hvd.init()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = str(hvd.local_rank())
            K.set_session(tf.Session(config=config))
    else:
        hvd = None

    logging.info("Running with MPI={}".format(mpi))
    logging.info("getting data")

    json_path = args.data_dir + '/dataset_info.json'

    with open(json_path, 'r') as f:
        info = json.load(f)

    args.test_num_examples = 0
    args.train_num_examples = 0

    for json_item in info['splits']:
        if json_item['name'] == 'test':
            args.test_num_examples = int(
                json_item['statistics']['numExamples'])
        elif json_item['name'] == 'train':
            args.train_num_examples = int(
                json_item['statistics']['numExamples'])

    STEPS_PER_EPOCH = args.train_num_examples // args.BATCH_SIZE
    VALIDATION_STEPS = args.test_num_examples//args.BATCH_SIZE//args.VAL_SUBSPLITS

    args.total_step = STEPS_PER_EPOCH * args.EPOCHS

    logging.info("STEPS_PER_EPOCH : {}".format(STEPS_PER_EPOCH))
    logging.info("VALIDATION_STEPS : {}".format(VALIDATION_STEPS))
    logging.info("args.train_num_examples : {}".format(
        args.train_num_examples))
    logging.info("args.BATCH_SIZE : {}".format(args.BATCH_SIZE))
    logging.info("args.test_num_examples : {}".format(args.test_num_examples))
    logging.info("args.VAL_SUBSPLITS : {}".format(args.VAL_SUBSPLITS))

#     dataset = _input()
    train_dataset = train_input_fn(args)
    test_dataset = test_input_fn(args)

    callbacks = [
        DisplayCallback(),
        ReduceLROnPlateau(),
        # tf.keras.callbacks.EarlyStopping(patience=30, verbose=1),
        # tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    ]

    if mpi:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
            warmup_epochs=5, verbose=1))
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            patience=10, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(
                args.output_data_dir + '/checkpoint-{epoch}.h5'))
            callbacks.append(tf.keras.callbacks.TensorBoard(
                args.output_data_dir, histogram_freq=1))
    else:
        callbacks.append(ModelCheckpoint(args.output_data_dir +
                                         '/checkpoint-{epoch}.h5', verbose=1, save_best_only=True, save_weights_only=True))
        callbacks.append(tf.keras.callbacks.TensorBoard(
            args.output_data_dir, histogram_freq=1))

    logging.info("Starting training")

    size = 1
    if mpi:
        size = hvd.size()

    model = model(args)

    model_history = model.fit(train_dataset, epochs=args.EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              callbacks=callbacks
                              )

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    logging.info('Train loss:{}'.format(loss))
    logging.info('Val loss:{}'.format(val_loss))

    if mpi:
        if hvd.rank() == 0:
            save_history(os.path.join(args.output_data_dir,
                                      'model_history.p'), model_history)
            save_model(model, args.artifact_dir)
    else:
        save_history(os.path.join(args.output_data_dir,
                                  'model_history.p'), model_history)
        save_model(model, args.artifact_dir)

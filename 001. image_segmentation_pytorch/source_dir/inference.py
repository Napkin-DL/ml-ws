# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
# language governing permissions and limitations under the License.


import io
import json
import numpy as np
from collections import namedtuple
from PIL import Image

import tensorflow as tf
import requests
import base64
import boto3

import os

import cv2


IMG_HEIGHT = 64
IMG_WIDTH = 64

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == 'application/x-image':
        sample_img = str(data.read(), 'utf-8')
        sample_img = Image.open(io.BytesIO(base64.decodebytes(sample_img.encode('utf-8'))))    
        sample_img = cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR)
        sample_img = cv2.resize(sample_img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        sample_img = np.float32(sample_img)
        sample_img = np.expand_dims(sample_img, axis=0)
        sample_img = sample_img / 255.0 

        return json.dumps({"instances": sample_img.tolist()})

    else:
        _return_error(415, 'Unsupported content type "{}"'.format(
            context.request_content_type or 'Unknown'))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise Exception(data.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = data.content
    prediction = json.loads(prediction)['predictions']

    pred_mask=np.array(prediction)
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., np.newaxis]
    pred_mask = pred_mask[0]
    
    success, pred_mask = cv2.imencode('.png', pred_mask)
    pred_mask = pred_mask.tobytes()
    file_byte_string = base64.encodebytes(pred_mask).decode("utf-8")
    
    return json.dumps(file_byte_string), response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))



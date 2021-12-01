# -*- coding: utf-8 -*-
"""ML zoomcamp week-9 homework.ipynb
"""

import numpy as np
import tflite_runtime.interpreter as tflite
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

interpreter = tflite.Interpreter(model_path="cats-dogs-v2.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


classes = [
    'cat',
    'dog'
]


def predict(url):
    image = download_image(url)
    image = prepare_image(image, (150, 150))

    x = np.array(image, dtype='float32')
    X = np.array([x])

    # Preprocess the image
    X = np.float32(X*(1./255))

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()
    print(float_predictions)

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

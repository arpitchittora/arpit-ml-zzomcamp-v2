# -*- coding: utf-8 -*-
"""ML zoomcamp week-9 homework.ipynb
"""

import numpy as np
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

from io import BytesIO
from urllib import request

from PIL import Image

interpreter = tflite.Interpreter(model_path="classification-model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(224, 224))

classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
           'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()
    print(float_predictions)

    result = dict(zip(classes, float_predictions))
    veg_type = max(zip(result.values(), result.keys()))[1]
    return veg_type


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

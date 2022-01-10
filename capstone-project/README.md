# Fruit and Vegetable Image Recognition (Capstone Project)

## Inspiration

The idea was to build an application that recognizes the food item(s) from the captured photo and gives its user different recipes that can be made using the food item(s).

## Context

[Kaggle dataset link](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition)

This dataset contains images of the following food items:

- fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.
- vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.

## This dataset contains three folders:

train (100 images each)

test (10 images each)

validation (10 images each)

each of the above folders contains subfolders for different fruits and vegetables wherein the images for respective food items are present

## Data Collection
The images in this dataset were scraped by me from Bing Image Search for a project of mine.

# How to run the project

Before running this project, You need to download datasets from [Kaggle datasets](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition).

The data size is 2GB so I didn't commit. You can download it from kaggle.

### How to run

[Capstone notebook link](https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/capstone-project/Capstone_Project_ML_zoomcamp.ipynb)

I am using Google Colab to run this project. You can run on your system if compatible or you can directly run google colab, Kaggle, or any other platform.

Firstly, you need kaggle (JSON file) file to download the dataset. Once data is downloaded then you can run each cell one by one.

We have different files in datasets file JPG or PNG. I am using only JPG files due to RGB color mode.

After running the above notebook, I saved the best model but we can't deploy this model because we are using TensorFlow so I converted it to TensorFlow lite version. To convert this model, I am using another notebook.

### Convert Tensorflow to TF-Lite

[Notebook link](https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/capstone-project/Capstone_lite_model_ipyn.ipynb)

In this notebook, I converted TensorFlow to TensorFlow lite version because we don't want to add TensorFlow on the server due to package size. For this, I used "keras-image-helper" and "tflite_runtime" package. Check the above notebook.

# How to deploy

Once we convert Tensorflow to Tensorflow lite version. I have created a Docker file to build an image and deploy it on the server.
Check [Docker file](https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/capstone-project/Dockerfile)

After that, we have created a python script file to run this model. Check [labda function file](https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/capstone-project/lambda_function.py).

In this file, we have created a lambda handler function to predict fruit and vegetables.

### Test model

To test our final model, I have created a [Test file](https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/capstone-project/test.py). You need to run this file. I have already attached my deployment link. This is working fine. It will give you the name of the fruit or vegetable. You don't need to deploy a model to test this file.

### Running lambda function

This model is running using lambda function. You only need to deploy code on EC2 machine and create a docker image after that you need to create a repository using Amazon Container Services. 

In this repository, You need to push your docker image using AWS ecr. After that, This repository can link to lanbda function and that lambda function will be using in API gateway to expose our lambda function or we can see that everyone can use it using web api that's why we AWS API Gateway and deploy it.

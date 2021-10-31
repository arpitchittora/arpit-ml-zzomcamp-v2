# IMDB rating prediction (ML Zoomcamp Midterm Project)

This project is about IMDB most popular movie or series based on IMDB rating. IMDB is a popular website for rating films and series I always go there if I want to watch something new and many many users trust it's rankings
The data is about most popular 7k Films and series on IMDB with rates, The Data is Ideal for Exploratory Data Analysis and I use Regression to predict Rating.

https://www.kaggle.com/mazenramadan/imdb-most-popular-films-and-series

In this project, I have tried multiple model to get best rating with better performance. Like
  - Linear regression
  - Decision Tree regression
  - Random forest regression
  - xgboost regression

I have found that Xgboost is better than other model. I have done parameter tunning in this model like eta, max_depth and min_child_weight.

# Model deployment
  
  I have deploy my model on AWS. You can check <a href="https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/mid-term-project/predict-test.py">test file.</a> In this file you can get api path and how to check model with new data.
  
  
# Run project on local
  
  If you want to run this project on lcoal. You can directly run <a href="https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/mid-term-project/predict-test.py">App file.</a>
  
  Or
  
  If you want to check notebook then we have 2 notebooks here. 
  
  - <a href="https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/mid-term-project/IMDB-Rating-notebook.ipynb"> IMDB-Rating-notebook </a>
      In this notebook, I have done all the things here like data cleaning, tried multiple models and other things.
      
  - <a href="https://github.com/arpitchittora/arpit-ml-zzomcamp-v2/blob/master/mid-term-project/IMDB-Rating-notebook.ipynb"> Final notebook </a>
     This notebook is final after model selection and I love xgboost and their better result.
     
  And also, I have deployed code on AWS using docker and conda venv. You can check this repository.


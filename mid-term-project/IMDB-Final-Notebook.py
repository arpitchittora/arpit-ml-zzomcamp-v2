#!/usr/bin/env python
# coding: utf-8

# import basic library
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import r2_score
import xgboost as xgb

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('imdb.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')


df.loc[df.rate == 'No Rate', 'rate'] = 0
df.rate = pd.to_numeric(df['rate'])


df.loc[df.votes == 'No Votes', 'votes'] = 0
df.votes = pd.to_numeric(df.votes.str.replace(",", ""), downcast="integer")

df.loc[df.duration == 'None', 'duration'] = 0
df.duration = pd.to_numeric(df.duration)

categorical_cols = ['genre', 'type', 'certificate', 'nudity',
                    'violence', 'profanity', 'alcohol', 'frightening']
numeric_cols = ['rate', 'votes', 'duration']


imdb_data = df[categorical_cols+numeric_cols].copy()


for col in categorical_cols:
    imdb_data[col] = imdb_data[col].str.lower()

imdb_data.votes = imdb_data.votes.fillna(imdb_data.votes.mean())

imdb_data.genre = imdb_data.genre.str.replace(' ', '')
genre_cols = imdb_data.genre.str.get_dummies(sep=',')

data = pd.concat([imdb_data, genre_cols], axis=1, join='inner')

del data['genre']


data.votes = data.votes.astype(int)


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


df_train, df_test = train_test_split(data, test_size=0.2, random_state=1)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.rate.values)
y_test = np.log1p(df_test.rate.values)

del df_train['rate']
del df_test['rate']


dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_test.to_dict(orient='records')
X_test = dv.transform(val_dict)


features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_test, label=y_test, feature_names=features)

xgb_params = {
    'eta': 0.3,
    'max_depth': 8,
    'min_child_weight': 2,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain, num_boost_round=100)
y_pred = model.predict(dval)
score_mean = rmse(y_test, y_pred)
xgb_r2_score = r2_score(y_test, y_pred)

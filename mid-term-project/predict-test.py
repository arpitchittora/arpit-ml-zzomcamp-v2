#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://imdb-rating-env.eba-z4c2hmhp.ap-south-1.elasticbeanstalk.com/predict'

movie_data = {
    "type": "film",  # series or film
    # 'pg-13', 'r', 'tv-ma', 'none', 'tv-14', 'tv-pg', 'pg', 'tv-g', '(banned)', 'not rated', 'e', 'nc-17', 'tv-y7-fv', 'tv-y7', 'unrated', 'approved', 'g', 'tv-y', 'gp', 'passed', 'm', 'x', 'm/pg'
    "certificate": "r",
    "nudity": "mild",  # 'mild', 'none', 'moderate', 'no rate', 'severe'
    "violence": "severe",  # 'mild', 'none', 'moderate', 'no rate', 'severe'
    "profanity": "severe",  # 'mild', 'none', 'moderate', 'no rate', 'severe'
    "alcohol": "mild",  # 'mild', 'none', 'moderate', 'no rate', 'severe'
    "frightening": "moderate",  # 'mild', 'none', 'moderate', 'no rate', 'severe'
    "votes": 203578,  # integer type
    "duration": 105,  # integer type in minutes
    "action": 1,  # 0 or 1
    "adventure": 0,  # 0 or 1
    "animation": 0,  # 0 or 1
    "biography": 0,  # 0 or 1
    "comedy": 0,  # 0 or 1
    "crime": 0,  # 0 or 1
    "documentary": 0,  # 0 or 1
    "drama": 0,  # 0 or 1
    "family": 0,  # 0 or 1
    "fantasy": 0,  # 0 or 1
    "film-noir": 0,  # 0 or 1
    "game-show": 0,  # 0 or 1
    "history": 0,  # 0 or 1
    "horror": 0,  # 0 or 1
    "music": 0,  # 0 or 1
    "musical": 0,  # 0 or 1
    "mystery": 0,  # 0 or 1
    "news": 0,  # 0 or 1
    "reality-tv": 0,  # 0 or 1
    "romance": 0,  # 0 or 1
    "sci-fi": 1,  # 0 or 1
    "short": 0,  # 0 or 1
    "sport": 0,  # 0 or 1
    "talk-show": 0,  # 0 or 1
    "thriller": 1,  # 0 or 1
    "war": 0,  # 0 or 1
    "western": 0  # 0 or 1
}


response = requests.post(url, json=movie_data).json()
print(response)

print('Movie rating would be %.3f' % response['expected_rating'])

import requests

url = "http://localhost:9696/predict"
customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}

return requests.post(url, json=customer).json()


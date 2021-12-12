import requests

url = 'https://ummkhnhndc.execute-api.ap-south-1.amazonaws.com/fruit-vegetable-indentification/predict'

data = {"url": "https://upload.wikimedia.org/wikipedia/commons/7/7d/Corncobs.jpg"}

result = requests.post(url, json=data).json()
print(result)

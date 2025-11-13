import requests

# URL zur deployed API (später ersetzen mit echter URL!)
url = "https://ml-pipeline-production.onrender.com/predict"

data = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

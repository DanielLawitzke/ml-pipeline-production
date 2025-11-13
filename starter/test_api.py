from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    """Test GET on root returns welcome message and 200 status"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML model API"}


def test_post_predict_high_income():
    """Test POST predicts >50K for high earner profile"""
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

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_post_predict_low_income():
    """Test POST predicts <=50K for low earner profile"""
    data = {
        "age": 22,
        "workclass": "Private",
        "fnlgt": 201490,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States",
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"

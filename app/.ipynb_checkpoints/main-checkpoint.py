import pickle
import logging
import google.cloud.logging
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# This line sets up the logging client
client = google.cloud.logging.Client()
client.setup_logging()

app = FastAPI(title="Iris ML API")
Instrumentator().instrument(app).expose(app)
# Load the model
with open('model/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post('/predict')
def predict_iris(iris_input: IrisInput):
    logging.info(f"Received prediction request: {iris_input.dict()}")
    input_data = np.array([[
        iris_input.sepal_length,
        iris_input.sepal_width,
        iris_input.petal_length,
        iris_input.petal_width
    ]])
    prediction = model.predict(input_data)
    result = {"prediction": int(prediction[0])}
    logging.info(f"Prediction result: {result}")
    return result
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Iris ML API")

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
    input_data = np.array([[
        iris_input.sepal_length,
        iris_input.sepal_width,
        iris_input.petal_length,
        iris_input.petal_width
    ]])
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
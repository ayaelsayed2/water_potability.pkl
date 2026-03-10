from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load model
model = joblib.load("../water_model.pkl")

@app.get("/")
def home():
    return {"message": "Water Quality Prediction API"}


@app.post("/predict")
def predict(
    ph: float,
    hardness: float,
    solids: float,
    chloramines: float,
    sulfate: float,
    conductivity: float,
    organic_carbon: float,
    turbidity: float
):

    data = np.array([[ph, hardness, solids, chloramines,
                      sulfate, conductivity, organic_carbon, turbidity]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Drinkable Water"
    else:
        result = "Not Drinkable Water"

    return {"prediction": result}
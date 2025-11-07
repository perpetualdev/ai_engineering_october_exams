# API Endpoint for Wine Prediction
# Import Libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Let us load our saved model
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Let us initialize our application
app = FastAPI()

# Create our pydantic model for validation
class WineFeatures(BaseModel):
  fixed_acidity: float
  volatile_acidity: float
  citric_acid: float
  residual_sugar: float
  chlorides: float
  free_sulfur_dioxide: float
  total_sulfur_dioxide: float
  density: float
  pH: float
  sulphates: float
  alcohol: float

# Create endpoints
# Root endpoint
@app.get('/')
def wine_home():
  return {"message": "Welcome to the Wine Quality Prediction App...🍷🍷🍷"}

# Prediction endpoint
@app.post("/predict")
def predict(wine: WineFeatures):
  try:
    # convert the features to a 2D numpy array using [[]]
    features = np.array([[
      wine.fixed_acidity,
      wine.volatile_acidity,
      wine.citric_acid,
      wine.residual_sugar,
      wine.chlorides,
      wine.free_sulfur_dioxide,
      wine.total_sulfur_dioxide,
      wine.density,
      wine.pH,
      wine.sulphates,
      wine.alcohol, 
    ]])

    # Scale our input features using the loaded scaler (to normalize the input)
    scaled_features = scaler.transform(features)

    # Make prediction with the loaded model
    prediction = model.predict(scaled_features)

    # Return the prediction(converted to string for serialization)
    return {"predicted_quality": str(prediction[0])}

    # Run prediction with "uvicorn main:app --reload"
  except Exception as e:
    raise HTTPException (status_code=500, detail=f"Error Predicting: {str(e)}")

import os
import joblib
import numpy as np
from sqlalchemy import Column,Integer,Float,String
from database import Base


# Load the model, scaler, and encoder
model = joblib.load("uber_trip.pkl")
scaler = joblib.load("scaler.pkl")
le_dispatching = joblib.load("label_encoder.pkl")


# Available dispatch options for dropdown
dispatching_options = le_dispatching.classes_.tolist()

def predict_trips(dispatching_base_number, active_vehicles, day, month, year):
    """Predict number of trips using trained model."""
    try:
        encoded_dispatch = le_dispatching.transform([dispatching_base_number])[0]
        features = np.array([[encoded_dispatch, active_vehicles, day, month, year]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        return round(float(prediction), 2)
    except Exception as e:
        return f"Error: {e}"

class Prediction(Base):
     __tablename__ = "predictions"

     id =Column(Integer,primary_key=True,index=True)
     dispatching_base_number = Column(Integer,index=True)
     active_vehicles = Column(Float)
     day = Column(Integer)
     month = Column(Integer)
     year=Column(Integer)
     prediction =Column(Float)


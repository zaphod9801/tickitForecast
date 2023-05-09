from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.models.LinearRegression import prediction_linear_regression
from app.models.Sarimax import prediction_sarimax

app = FastAPI()


@app.get("/predict/")
async def prediccion():
    try:
        best_preds, best_score, best_params = prediction_sarimax()
        predictionSarimax = best_preds.tolist()

        predictionLinear, mse = prediction_linear_regression()
        predictionLinear = predictionLinear.tolist()
        
        response = {
        
          "Sarimax": {
              "prediction by Sarimax": predictionSarimax,
              "mse": best_score,
              "Parameters": best_params
          },
          
          "Linear Regression": {
              "prediction by Linear Regression": predictionLinear,
              "mse": mse
          }
    
        }
        
        
        return response
    except Exception as e:
        return {"error": str(e)}


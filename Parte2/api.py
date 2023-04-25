import pickle
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, validator
import numpy as np
from sklearn.preprocessing import StandardScaler


class InputData(BaseModel):
    age: int = Field(..., description="Edad de la persona (1 a 99)")
    children: int = Field(..., description="Numero de hijos (0 a 15)")
    smoker_yes: int = Field(..., description="Fumador (0 - no, 1 - si)")
    region_northeast: int = Field(..., description="Vive en la region noreste (0 - no, 1 - si)")
    region_northwest: int = Field(..., description="Vive en la region noroeste (0 - no, 1 - si)")
    region_southwest: int = Field(..., description="Vive en la region suroeste (0 - no, 1 - si)")
    region_southeast: int = Field(..., description="Vive en la region sureste (0 - no, 1 - si)")
    overweight: int = Field(..., description="Sobrepeso (IMC > 30) (0 - no, 1 - si) ")
    unhealthy: int = Field(..., description="Fumador y con sobrepreso")

  # Define validators for the input data fields
    @validator('age')
    def check_age(cls, value):
        if value < 0. or value > 99:
            raise ValueError("La edad debe estar entre 0 y 99")
        return value

    @validator('children')
    def check_children(cls, value):
        if value < 0 or value > 15:
            raise ValueError("El numero de hijos debe estar entre 0 y 15")
        return value
   
    @validator('smoker_yes')
    def check_smoker(cls, value):
        if value < 0 or value > 1:
            raise ValueError("El valor debe ser 0 (no fumador) o 1 (fumador)")
        return value

    @validator('region_northeast')
    def check_northeast(cls, value):
        if value < 0 or value > 1:
            raise ValueError("El valor debe ser 0 (no habita ahi) o 1 (habita)")
        return value

    @validator('region_northwest')
    def check_northwest(cls, value):
        if value < 0 or value > 1:
            raise ValueError("El valor debe ser 0 (no habita ahi) o 1 (habita)")
        return value

    @validator('region_southwest')
    def check_southwest(cls, value):
        if value < 0 or value > 1:
            raise ValueError("El valor debe ser 0 (no habita ahi) o 1 (habita)")
        return value

    @validator('region_southeast')
    def check_southeast(cls, value):
        if value < 0 or value > 1:
            raise ValueError("El valor debe ser 0 (no habita ahi) o 1 (habita)")
        return value
   
    @validator('overweight')
    def check_overweight(cls, value):
        if value < 0 or value > 1:
            raise ValueError("El valor debe ser 0 (sin sobrepeso) o 1 (con sobrepeso)")
        return value
   
    @validator('unhealthy')
    def check_unhealthy(cls, value):
        if value < 0 or value > 1:
            raise ValueError("El valor debe ser 0 (sano) o 1 (no sano)")
        return value

# Load the pre-trained regression model
with open("lasso_opt_model.pkl", "rb") as f:
    model = pickle.load(f)


# Create the FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Bienvenidos a la API para predecir precios de aseguranza medica en Estados Unidos!"}

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    input_dict = input_data.dict()
    # Convert input data to a 2D numpy array for prediction
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    #scaler = StandardScaler()
    #input_array = scaler.fit_transform(input_array)
   
    # Use the pre-trained model to make predictions
    prediction = model.predict(input_array)[0]
   
    # Return the prediction as a JSON response
    return {"precio de la aseguranza": f"${np.round(prediction.item(), 2)}"}
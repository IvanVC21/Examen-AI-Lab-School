import pickle
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, validator
import numpy as np
from sklearn.preprocessing import StandardScaler


class InputData(BaseModel):
    carat: float = Field(..., description="The weight of the diamond in carats (0.0 to 5.0)")
    cut: int = Field(..., description="The quality of the diamond's cut (1 to 5)")
    color: int = Field(..., description="The color of the diamond (1 to 7)")
    clarity: int = Field(..., description="The clarity of the diamond (1 to 8)")
    x: float = Field(..., description="The length of the diamond in mm (0.0 to 11.0)")
    y: float = Field(..., description="The width of the diamond in mm (0.0 to 60.0)")
    z: float = Field(..., description="The depth of the diamond in mm (0.0 to 35.0)")
    depth: int = Field(..., description="The depth percentage of the diamond (40 to 80)")
    table: int = Field(..., description="The width of the top of the diamond relative to its widest point (40 to 100)")

  # Define validators for the input data fields
    @validator('carat')
    def check_carat(cls, value):
        if value < 0.0 or value > 5.0:
            raise ValueError("must be between 0.0 and 5.0")
        return value

    @validator('cut')
    def check_cut(cls, value):
        if value < 1 or value > 5:
            raise ValueError("must be between 1 and 5")
        return value
    
    @validator('color')
    def check_color(cls, value):
        if value < 1 or value > 7:
            raise ValueError("must be between 1 and 7")
        return value

    @validator('clarity')
    def check_clarity(cls, value):
        if value < 1 or value > 8:
            raise ValueError("must be between 1 and 8")
        return value

    @validator('x')
    def check_x(cls, value):
        if value < 0.0 or value > 11.0:
            raise ValueError("must be between 0.0 and 11.0")
        return value

    @validator('y')
    def check_y(cls, value):
        if value < 0.0 or value > 60.0:
            raise ValueError("must be between 0.0 and 60.0")
        return value

    @validator('z')
    def check_z(cls, value):
        if value < 0.0 or value > 35.0:
            raise ValueError("must be between 0.0 and 35.0")
        return value
    
    @validator('depth')
    def check_depth(cls, value):
        if value < 40 or value > 80:
            raise ValueError("must be between 40 and 80")
        return value
    
    @validator('table')
    def check_table(cls, value):
        if value < 40 or value > 100:
            raise ValueError("must be between 40 and 100")
        return value

# Load the pre-trained regression model 
with open("modelo_examen_parte1.pkl", "rb") as f:
    model = pickle.load(f)


# Create the FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Bienvenidos a la API para predecir precios de diamantes!"}

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
    return {"precio": f"${np.round(prediction.item(), 2)}"}



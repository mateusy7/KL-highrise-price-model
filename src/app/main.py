from fastapi import FastAPI
from src.model.inference import ModelInference
from pydantic import BaseModel

app = FastAPI()
model = ModelInference()

class HighriseData(BaseModel):
    property_type: str
    mukim: str
    scheme_name_area:str
    tenure: str

    land_parcel_area: float
    unit_level: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Kuala Lumpur Highrise Price Model is Live"}

@app.post("/predict")
def get_prediction(data: HighriseData):
    
    query = data.dict()

    prediction = model.predict(query)

    return {"estimated_price_rm":prediction[0].round()}
from typing import List
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app=FastAPI()

LABELS=[
    'Verdante',
    'Rubresco',
    'Floralis'
]

class Features(BaseModel):
    features: List[float]

async def load_model(file_path):
    return joblib.load(filename=file_path)


@app.post(path="/predict",status_code=status.HTTP_200_OK)
async def predict(features: Features):
    try:
        model=await load_model(file_path="artifacts/model.pkl")
        prediction_index=model.predict([features.features])[0]
        prediction_label=LABELS[prediction_index]
        return  {'prediction':prediction_label}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

if  __name__=="__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=9000)


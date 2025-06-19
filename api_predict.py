from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Cargar modelos y encoders
clf_type = joblib.load('model_resource_type.pkl')
clf_format = joblib.load('model_resource_format.pkl')
encoders = joblib.load('encoders.pkl')

# Features esperadas
features = [
    'university_id', 'carreer', 'module_id', 'module_title', 'roadmap_id',
    'roadmap_title', 'roadmap_level', 'roadmap_category_id', 'country', 'city',
    'type_assessment_id', 'module_skills', 'module_objectives',
    'previous_resource_type', 'previous_resource_format',
    'resource_success_rate', 'resource_popularity'
]

class PredictionInput(BaseModel):
    university_id: str
    carreer: str
    module_id: int
    module_title: str
    roadmap_id: int
    roadmap_title: str
    roadmap_level: str
    roadmap_category_id: int
    country: str
    city: str
    type_assessment_id: int
    module_skills: str
    module_objectives: str
    previous_resource_type: str
    previous_resource_format: str
    resource_success_rate: float
    resource_popularity: int
    age: int
    education_level: int
    learning_style: int
    prior_experience: int

@app.post("/predict")
def predict(input: PredictionInput):
    # Convertir input a DataFrame
    input_dict = input.dict()
    df = pd.DataFrame([input_dict])
    # Codificar igual que en entrenamiento
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
    # Predecir
    pred_type = clf_type.predict(df)[0]
    pred_format = clf_format.predict(df)[0]
    # Decodificar
    type_label = encoders['target_resource_type'].inverse_transform([pred_type])[0]
    format_label = encoders['target_resource_format'].inverse_transform([pred_format])[0]
    return {
        "predicted_resource_type": type_label,
        "predicted_resource_format": format_label
    }

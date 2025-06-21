
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Cargar modelos y codificadores
model = joblib.load("resource_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
output_classes = joblib.load("output_classes.pkl")

class ResourceInput(BaseModel):
    module_title: str
    roadmap_title: str
    roadmap_level: int
    avg_score: int
    type_assessment: str
    categories: str  # no usado en predicción pero recibido
    formats: str     # no usado en predicción pero recibido
    total_modules_done: int

@app.post("/predict")
def predict(input: ResourceInput):
    input_data = {
        "module_title": label_encoders["module_title"].transform([input.module_title])[0],
        "roadmap_title": label_encoders["roadmap_title"].transform([input.roadmap_title])[0],
        "roadmap_level": input.roadmap_level,
        "avg_score": input.avg_score,
        "type_assessment": label_encoders["type_assessment"].transform([input.type_assessment])[0],
        "total_modules_done": input.total_modules_done
    }

    X = [[
        input_data["module_title"],
        input_data["roadmap_title"],
        input_data["roadmap_level"],
        input_data["avg_score"],
        input_data["type_assessment"],
        input_data["total_modules_done"]
    ]]

    prediction = model.predict(X)[0]
    n_cat = len(output_classes["categories"])
    cat_preds = [output_classes["categories"][i] for i, val in enumerate(prediction[:n_cat]) if val == 1]
    fmt_preds = [output_classes["formats"][i] for i, val in enumerate(prediction[n_cat:]) if val == 1]

    return {
        "predicted_categories": cat_preds,
        "predicted_formats": fmt_preds
    }

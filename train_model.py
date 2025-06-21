
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import joblib

# Cargar dataset
df = pd.read_csv("training_dataset.csv")

# Preprocesamiento de multilabels
def split_multilabel(col):
    return [item.strip() for item in col.split(',') if item.strip()]

df["categories"] = df["categories"].apply(split_multilabel)
df["formats"] = df["formats"].apply(split_multilabel)

# Codificación multilabel
mlb_categories = MultiLabelBinarizer()
mlb_formats = MultiLabelBinarizer()
y_categories = mlb_categories.fit_transform(df["categories"])
y_formats = mlb_formats.fit_transform(df["formats"])
y = np.hstack([y_categories, y_formats])

# Features
X = df[["module_title", "roadmap_title", "roadmap_level", "avg_score", "type_assessment", "total_modules_done"]].copy()
label_encoders = {}
for col in ["module_title", "roadmap_title", "type_assessment"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Guardar modelos
joblib.dump(model, "resource_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump({
    "categories": mlb_categories.classes_.tolist(),
    "formats": mlb_formats.classes_.tolist()
}, "output_classes.pkl")

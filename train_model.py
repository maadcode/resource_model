import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Cargar datos
file_path = 'data_entrenamiento_full.csv'  # Corregido: ruta relativa correcta
data = pd.read_csv(file_path)

# Columnas de entrada (features)
features = [
    'university_id', 'carreer', 'module_id', 'module_title', 'roadmap_id',
    'roadmap_title', 'roadmap_level', 'roadmap_category_id', 'country', 'city',
    'type_assessment_id', 'module_skills', 'module_objectives',
    'previous_resource_type', 'previous_resource_format',
    'resource_success_rate', 'resource_popularity'
]

# Columnas objetivo
label_type = 'target_resource_type'
label_format = 'target_resource_format'

# Codificar variables categóricas
encoders = {}
for col in features + [label_type, label_format]:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

# Separar features y labels
X = data[features]
y_type = data[label_type]
y_format = data[label_format]

# Dividir en train/test
X_train, X_test, y_type_train, y_type_test, y_format_train, y_format_test = train_test_split(
    X, y_type, y_format, test_size=0.2, random_state=42
)

# Entrenar modelos
clf_type = RandomForestClassifier(n_estimators=100, random_state=42)
clf_type.fit(X_train, y_type_train)

clf_format = RandomForestClassifier(n_estimators=100, random_state=42)
clf_format.fit(X_train, y_format_train)

# Guardar modelos y encoders
joblib.dump(clf_type, 'model_resource_type.pkl')
joblib.dump(clf_format, 'model_resource_format.pkl')
joblib.dump(encoders, 'encoders.pkl')

print('Modelos entrenados y guardados en la carpeta Modelo.')

"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Inicializamos la app
app = FastAPI(title="API de Predicción de Precios de Vivienda (California)", version="1.0")

# Esquema de datos esperado por la API
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # Categórica: INLAND, NEAR BAY, NEAR OCEAN, <1H OCEAN, ISLAND

# Variable global para cargar el modelo
model = None

@app.on_event("startup")
def load_model():
    """
    Carga el modelo globalmente al iniciar el servidor usando joblib.
    """
    global model
    try:
        model = joblib.load("models/model.pkl")
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"⚠️ No se pudo cargar el modelo: {e}")

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos"}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Recibe las características de una vivienda y retorna el precio predicho.
    """
    if model is None:
        return {"error": "El modelo no se ha cargado."}

    # 1. Convertir a DataFrame con las mismas columnas del entrenamiento
    data = pd.DataFrame([{
        "longitude":          features.longitude,
        "latitude":           features.latitude,
        "housing_median_age": features.housing_median_age,
        "total_rooms":        features.total_rooms,
        "total_bedrooms":     features.total_bedrooms,
        "population":         features.population,
        "households":         features.households,
        "median_income":      features.median_income,
        "ocean_proximity":    features.ocean_proximity
    }])

    # 2. Predecir
    prediction = model.predict(data)[0]

    # 3. Retornar resultado
    return {
        "predicted_price": round(float(prediction), 2),
        "mensaje": f"El precio estimado de la vivienda es ${prediction:,.2f} USD"
    }

# Instrucciones para correr la API localmente:
# 1. Ejecutar en terminal: uvicorn src.api.main:app --reload
# 2. Pegar: http://127.0.0.1:8000/docs en el buscador
# En la API, pega los datos de la primera fila del dataset para comparar, cuyo precio real es $452,600, El modelo predijo $448,568.
# 

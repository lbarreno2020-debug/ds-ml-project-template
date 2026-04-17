"""
Script para descargar y extraer los datos originales del proyecto.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

def fetch_housing_data(housing_url: str, housing_path: str):
    """
    INSTRUCCIONES:
    1. Asegúrate de que el directorio `housing_path` exista (usa os.makedirs o Path.mkdir).
    2. Usa urllib.request.urlretrieve para descargar el archivo .tgz desde `housing_url`.
    3. Usa tarfile.open para extraer el contenido en `housing_path`.
    
    URL de los datos: "https://github.com/ageron/data/raw/main/housing.tgz"
    Ruta de destino recomendada: "data/raw/"
    """
    # 1. Crear el directorio si no existe
    os.makedirs(housing_path, exist_ok=True)

    # 2. Descargar el archivo .tgz
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    print("✅ Datos descargados correctamente")

    # 3. Extraer el contenido
    with tarfile.open(tgz_path) as f:
        f.extractall(housing_path)
    print("✅ Datos extraídos en:", housing_path)

    pass

if __name__ == "__main__":
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = "data/raw/"
    fetch_housing_data(URL, PATH)
    print("Script completado exitosamente!")

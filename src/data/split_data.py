"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    # 1. Leer los datos
    df = pd.read_csv(raw_data_path)
    print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 2. Crear categoría de ingreso para estratificar
    df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])
    
    # 3. Dividir con StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(df, df["income_cat"]):
        train_set = df.loc[train_idx].drop("income_cat", axis=1)
        test_set  = df.loc[test_idx].drop("income_cat", axis=1)

    # 4. Guardar los archivos
    Path(interim_data_path).mkdir(parents=True, exist_ok=True)
    train_set.to_csv(f"{interim_data_path}/train_set.csv", index=False)
    test_set.to_csv(f"{interim_data_path}/test_set.csv",  index=False)

    print(f"✅ Train: {train_set.shape[0]} filas → data/interim/train_set.csv")
    print(f"✅ Test:  {test_set.shape[0]} filas → data/interim/test_set.csv")

if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
    print("Split completado exitosamente!")

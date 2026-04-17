"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""

import pandas as pd
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def remove_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con inconsistencias lógicas detectadas en el EDA.
    """
    filas_antes = len(df)
    
    # Inconsistencia 1: population < households
    mask1 = df["population"] < df["households"]
    
    # Inconsistencia 2: rooms_per_household < 1
    mask2 = (df["total_rooms"] / df["households"]) < 1
    
    # Eliminar filas inconsistentes
    df = df[~(mask1 | mask2)]
    
    filas_despues = len(df)
    eliminadas = filas_antes - filas_despues
    
    print(f"✅ Filas eliminadas por inconsistencias: {eliminadas}")
    print(f"✅ Filas restantes: {filas_despues}")
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Maneja los valores faltantes.
       Puedes llenarlos con la mediana de la columna.
    2. Retorna el DataFrame limpio.
    """
    # Tu código aquí

    # Imputar valores nulos con la mediana
    imputer = SimpleImputer(strategy="median")
    cols_numericas = df.select_dtypes(include=[np.number]).columns
    df[cols_numericas] = imputer.fit_transform(df[cols_numericas])

    print(f"✅ Valores nulos después de limpieza: {df.isnull().sum().sum()}")

    return df



def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Agrega nuevas variables derivando las existentes, por ejemplo:
       - 'rooms_per_household' = total_rooms / households
       - 'population_per_household' = population / households
       - 'bedrooms_per_room' = total_bedrooms / total_rooms
    2. Retorna el DataFrame enriquecido.
    """
    # Tu código aquí

    # Nuevas variables combinadas
    df["rooms_per_household"]      = df["total_rooms"]    / df["households"]
    df["population_per_household"] = df["population"]     / df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]

    print("✅ Nuevas variables creadas: rooms_per_household, population_per_household, bedrooms_per_room")
    
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variables categóricas usando OneHotEncoder (Nominal).
    """
    if "ocean_proximity" in df.columns:
        ohe = OneHotEncoder(sparse_output=False)
        ocean_encoded = ohe.fit_transform(df[["ocean_proximity"]])
        ocean_df = pd.DataFrame(ocean_encoded,
                                columns=ohe.get_feature_names_out(["ocean_proximity"]),
                                index=df.index)
        df = df.drop("ocean_proximity", axis=1)
        df = pd.concat([df, ocean_df], axis=1)
        print("✅ Codificación OneHot aplicada a ocean_proximity")
    return df

def scale_features(df: pd.DataFrame, target_col: str = "median_house_value") -> pd.DataFrame:
    """
    Aplica StandardScaler a las variables numéricas excepto el target.
    """
    cols_escalar = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col]
    scaler = StandardScaler()
    df[cols_escalar] = scaler.fit_transform(df[cols_escalar])
    print("✅ StandardScaler aplicado a variables numéricas")
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que toma el DataFrame crudo y aplica limpieza y enriquecimiento.
    """
    df = remove_inconsistencies(df)
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)
    df = encode_categoricals(df)
    df = scale_features(df)
    
    # IMPORTANTE: Aquí los alumnos deberían añadir codificación de variables categóricas
    # (ej. get_dummies para 'ocean_proximity') si no usan Pipelines de Scikit-Learn.
    
    return df

if __name__ == "__main__":
    import os

    INTERIM_PATH = "data/interim/"
    os.makedirs(INTERIM_PATH, exist_ok=True)

    # Procesar Train
    df_train = pd.read_csv("data/interim/train_set.csv")
    print(f"✅ Train cargado: {df_train.shape[0]} filas x {df_train.shape[1]} columnas")
    df_train_processed = preprocess_pipeline(df_train)
    df_train_processed.to_csv(f"{INTERIM_PATH}/train_clean.csv", index=False)
    print(f"✅ train_clean.csv guardado: {df_train_processed.shape[1]} columnas")

    # Procesar Test
    df_test = pd.read_csv("data/interim/test_set.csv")
    print(f"\n✅ Test cargado: {df_test.shape[0]} filas x {df_test.shape[1]} columnas")
    df_test_processed = preprocess_pipeline(df_test)
    df_test_processed.to_csv(f"{INTERIM_PATH}/test_clean.csv", index=False)
    print(f"✅ test_clean.csv guardado: {df_test_processed.shape[1]} columnas")
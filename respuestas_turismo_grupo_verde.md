# Respuestas sustentadas - Predicción Turismo (Grupo Verde)

Fuente analizada: `src/PrediccionTurismo_GrupoVerde.ipynb` (ejecución completa validada).

## Resumen rápido (lectura no técnica)
- Sí hay suficiente información para predecir flujo de turistas por país de residencia.
- Se detectaron faltantes y valores extremos, pero se trataron en el flujo.
- Se entrenaron 4 modelos y se compararon con métricas estándar de regresión.
- El mejor en **test** por error cuadrático (RMSE) fue **Ridge**.
- En **validación cruzada**, el más estable por RMSE promedio fue **Gradient Boosting Regressor**.

## Variables de consulta final (usuario)
La salida final del modelo está pensada para consultar solo estas variables:
1. `Mes`
2. `País de origen/residencia`
3. `Tipo de visitante`
4. `Número de turistas` (predicción)

Nota: `Estado` se usa internamente para entrenar mejor el modelo, pero no se expone en la consulta final.

## Cantidades usadas en el entrenamiento
Para que se entienda exactamente con qué volumen se entrenó:
1. Dataset consolidado inicial: **18,120** filas.
2. Filas con `Turistas` reportado (sin nulo): **14,678**.
3. Filas finales para modelado (después de features de rezago): **10,400**.
4. Split de entrenamiento/prueba:
`Train = 9,865` filas y `Test = 535` filas (año de prueba: **2026**).
5. Validación adicional:
**KFold de 5 particiones** sobre el conjunto de entrenamiento.

**Fragmento de resultado (evidencia):**

```text
Forma del dataset consolidado: (18120, 8)
Registros listos para modelado: (10400, 11)
Año de test: 2026
Train: (9865, 8) | Test: (535, 8)
```

```python
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

## 1) ¿Qué problemas de calidad de datos se identificaron?

Se identificaron principalmente:
1. Valores faltantes en la variable objetivo (`Turistas`) por meses sin reporte en algunas hojas.
2. Valores extremos (outliers), esperables en destinos con alto volumen de turismo.
3. Diferencias de estructura por año/mes que obligan a consolidar hojas antes de modelar.

**Fragmento de resultado (evidencia):**

```text
Forma del dataset consolidado: (18120, 8)
Valores faltantes:
Turistas 3442 (19.0%)
```

```text
Q1=62.00 | Q3=1,360.75 | IQR=1,298.75
Outliers detectados por IQR: 1,829 (12.46% del total con Turistas no nulo)
```

## 2) ¿Qué técnicas de preprocesamiento fueron necesarias?

Se aplicaron técnicas básicas pero robustas:
1. Consolidación de hojas Excel por año en una sola tabla limpia.
2. Conversión de meses a número (`Mes_Num`) y construcción de fecha.
3. Tratamiento de faltantes con imputación dentro del pipeline.
4. Codificación de variables categóricas (`Pais`, `Estado`, `Tipo_Visitante`) con One-Hot para mejorar aprendizaje interno.
5. Escalamiento de variables numéricas.
6. Ingeniería de características temporales (`lag_1`, `lag_12`, `rolling_3_mean`).

**Fragmento de código (evidencia):**

```python
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
```

```python
df_model["lag_1"] = df_model.groupby(grupo)["Turistas"].shift(1)
df_model["lag_12"] = df_model.groupby(grupo)["Turistas"].shift(12)
df_model["rolling_3_mean"] = (
    df_model.groupby(grupo)["Turistas"]
    .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
)
```

## 3) ¿Qué variables fueron más importantes en el modelo?

Las variables más influyentes fueron las de comportamiento histórico reciente:
1. `lag_12` (valor de hace 12 meses).
2. `lag_1` (valor del mes anterior).
3. `rolling_3_mean` (promedio de corto plazo).
4. En menor medida, variables geográficas internas y tipo de visitante.

**Fragmento de resultado (evidencia):**

```text
Top 12 variables más importantes (Random Forest):
1) num__lag_12 (0.882066)
2) num__lag_1 (0.092401)
3) num__rolling_3_mean (0.019464)
4) num__Mes_Num (0.001828)
5) num__Anio (0.001118)
```

```text
Top coeficientes (Ridge):
1) num__lag_12
2) num__lag_1
3) cat__Estado_Quebec
4) cat__Estado_Ontario
5) num__rolling_3_mean
```

## 4) ¿Qué algoritmo tuvo mejor desempeño?

Depende del criterio:
1. **En test (RMSE):** mejor fue **Ridge**.
2. **En test (MAE):** menor error absoluto lo logró **Random Forest**.
3. **En validación cruzada (CV_RMSE):** mejor promedio lo logró **Gradient Boosting Regressor**.

Interpretación simple: Ridge rindió mejor en la partición final de prueba, pero Gradient Boosting fue más consistente cuando se re-entrena en varias particiones.

**Fragmento de resultado (evidencia):**

```text
Resultados en test:
Ridge                       MAE=407.43  RMSE=729.84  R2=0.9942
Linear Regression           MAE=407.62  RMSE=730.18  R2=0.9942
Random Forest Regressor     MAE=263.84  RMSE=811.86  R2=0.9928
Gradient Boosting Regressor MAE=296.64  RMSE=867.05  R2=0.9918

Mejor modelo en test (por RMSE): Ridge
```

```text
Validación cruzada (CV):
Gradient Boosting Regressor | CV_RMSE=1060.33 +/- 84.28 | CV_R2=0.9822
Random Forest Regressor     | CV_RMSE=1088.73 +/- 134.83 | CV_R2=0.9809
Ridge                       | CV_RMSE=1148.50 +/- 252.29 | CV_R2=0.9783
Linear Regression           | CV_RMSE=1149.15 +/- 252.98 | CV_R2=0.9783
```

## 5) ¿Cómo se comporta el modelo con nuevos datos?

El notebook sí demuestra predicción en registros nuevos. Internamente predice por estado (para mantener precisión), pero para usuario final entrega resultados agregados solo por:
1. Mes
2. País de origen/residencia
3. Tipo de visitante
4. Número de turistas

**Fragmento de resultado (evidencia):**

```text
Predicción agregada para: Marzo 2026

Mes    Pais            Tipo_Visitante  Prediccion_Turistas
marzo  Estados Unidos  Extranjero      259,750
marzo  Canadá          Extranjero      164,777
marzo  Estados Unidos  Dominicano       86,435
marzo  Resto           Extranjero       84,774
...

Consulta directa:
Predicción Marzo 2026 | Estados Unidos | Extranjero: 259,750 turistas
```

## 6) ¿Qué limitaciones tiene el dataset utilizado?

Limitaciones principales:
1. Serie histórica relativamente corta (2022-2026), lo que limita patrones de largo plazo.
2. Meses incompletos en 2026, que introducen faltantes en la variable objetivo.
3. El modelo aprende muy fuerte de rezagos; cambios abruptos (crisis, eventos, políticas) pueden reducir precisión.
4. El análisis es un baseline metodológicamente completo, pero sin tuning avanzado de hiperparámetros.

**Fragmento de resultado (evidencia):**

```text
Años disponibles: [2022, 2023, 2024, 2025, 2026]
Valores faltantes en Turistas: 3442 (19.0%)
Registros listos para modelado: (10400, 11)
```

## Cierre (en palabras simples)
El trabajo cumple la metodología pedida por la asignación: se exploró y limpió la data, se construyeron variables útiles, se compararon varios algoritmos con métricas claras, se hizo validación cruzada y se demostró predicción con datos nuevos. Para una audiencia no técnica, la conclusión es que el enfoque funciona y es defendible, con la precaución de que depende mucho del comportamiento histórico reciente.

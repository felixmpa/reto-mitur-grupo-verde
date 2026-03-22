# Predicción de Turismo por Estado de Residencia (Grupo Verde)

**Objetivo:** construir un modelo de regresión para predecir la cantidad de turistas por país/estado de residencia y tipo de visitante (Dominicano/Extranjero), cumpliendo todas las etapas metodológicas: exploración, preprocesamiento, entrenamiento, evaluación, validación y predicción en nuevos registros.

## Requisitos

- Python 3.10+
- `pip`

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar el notebook

```bash
jupyter notebook src/PrediccionTurismo_GrupoVerde.ipynb
```

## Notas

- KaggleHub usa cache local, por lo que no descarga en cada ejecucion.
- No subas `.env` con secretos reales a repositorios publicos.

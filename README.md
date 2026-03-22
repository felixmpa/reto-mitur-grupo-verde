# Asignacion Individual - Modelado Predictivo con Datos Abiertos

Proyecto para la asignacion individual de mineria de datos.

## Estado actual

- Notebook implementado: `src/titanic-clasification.ipynb`
- Notebook implementado: `src/house-prices-regression.ipynb`
- Carga automatica del dataset Titanic con `kagglehub`
- Carga automatica del dataset House Prices con `kagglehub` (competencia)
- Archivos de entorno en raiz: `.env` y `.env.example`

## Requisitos

- Python 3.10+
- `pip`

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Credenciales de Kaggle

El notebook usa `kagglehub.load_dataset(...)` y cache local de KaggleHub.

Configura `.env` en la raiz con:

```bash
cp .env.example .env
```

### Opcion recomendada (KaggleHub 1.0.0)

```env
KAGGLE_API_TOKEN=tu_api_token_de_kaggle
```

### Opciones alternativas

```env
KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_api_key_de_kaggle
```

O usar `~/.kaggle/kaggle.json`.

## Dataset de Titanic

Dataset: [yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

Comportamiento de carga del notebook:

1. Si existe `data/titanic/Titanic-Dataset.csv`, usa ese archivo local.
2. Si no existe, ejecuta `kagglehub.load_dataset(...)` para descargar/cargar el CSV.
3. Guarda una copia local en `data/titanic/Titanic-Dataset.csv`.

## Dataset de House Prices (Regresion)

Competencia: [house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Comportamiento de carga del notebook:

1. Si existe `data/house-prices/train.csv`, usa ese archivo local.
2. Si no existe, ejecuta `kagglehub.competition_download(...)` para descargar `train.csv`.
3. Guarda una copia local en `data/house-prices/train.csv`.

Importante:

- Antes de descargar por API debes aceptar las reglas de la competencia:
  `https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules`

## Ejecutar el notebook

```bash
jupyter notebook src/titanic-clasification.ipynb
jupyter notebook src/house-prices-regression.ipynb
```

## Notas

- KaggleHub usa cache local, por lo que no descarga en cada ejecucion.
- No subas `.env` con secretos reales a repositorios publicos.

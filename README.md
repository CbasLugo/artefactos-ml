# artefactos-ml

Artefactos entrenados de **TechMind AI**, para que la API los descargue al
arrancar sin cargarlos dentro de la imagen de Docker.

Proyecto: [G9-LATAM-Team-46](https://github.com/No-Country-simulation/G9-LATAM-Team-46)

| Archivo | Peso | Qué es |
|---|--:|---|
| `modelo_techmind_v2.joblib` | 5,9 MB | El clasificador: Pipeline de scikit-learn con TF-IDF + Regresión Logística |
| `matriz_historica.pkl` | 50,2 MB | Los 38.257 documentos del corpus vectorizados, para el contenido relacionado |

## El clasificador

`Pipeline` completo: recibe texto crudo y devuelve la predicción, con el
vectorizador adentro. Entrenado con `class_weight='balanced'`, `C=4.0`,
vocabulario de 60.000 términos, 8 categorías.

F1 macro 0,7549 sobre un test de 7.652 textos. Validación cruzada de 5
particiones: 0,7508 ± 0,0019.

## La matriz histórica

Diccionario con seis claves:

| Clave | Contenido |
|---|---|
| `vectorizador` | El `TfidfVectorizer` con el que se construyó la matriz |
| `matriz` | Matriz dispersa CSR de 38.257 × 20.000 |
| `ids` | Identificador de cada documento |
| `categorias` | Categoría de cada documento |
| `titulos` | Título de cada documento |
| `extractos` | Primeros 200 caracteres del cuerpo, cortados en el último espacio |

El vectorizador viaja adentro a propósito: un texto nuevo tiene que
vectorizarse con el mismo vocabulario con el que se construyó la matriz, o
los vectores no serían comparables.

No se guarda la matriz de similitudes entre todos los pares: con 38.257
documentos serían más de mil cuatrocientos millones de valores, casi todos
cercanos a cero. Se guardan los vectores y se compara contra ellos
únicamente el texto de la consulta.

## Cómo los consume la API

```
MODELO_URL=https://github.com/CbasLugo/artefactos-ml/raw/main/modelo_techmind_v2.joblib
MATRIZ_HISTORICA_URL=https://github.com/CbasLugo/artefactos-ml/raw/main/matriz_historica.pkl
```

## Compatibilidad de versiones

Los dos archivos son pickles creados con Python 3.12:

```
scikit-learn==1.8.0
numpy>=2.0.0
scipy>=1.13.0
```

La versión de scikit-learn no es negociable: el formato interno de
`TfidfVectorizer` y `LogisticRegression` cambia entre versiones, y cargarlos
con otra devuelve resultados distintos sin que nada falle a la vista. Numpy
va fijado porque los pickles llevan arreglos serializados con numpy 2.x, y
cargarlos con 1.x falla con `No module named 'numpy._core'`.

## Si se reentrena el modelo

Hay que regenerar los dos, no solo el `.joblib`: la matriz guarda su propio
vectorizador, y si el modelo cambia y ella no, quedan hablando de
vocabularios distintos.

Se generan con los notebooks de `machine_learning/` en el repositorio del
proyecto.

## Falta acá

`sugerencias_botones.json` (2 KB), que alimenta `GET /sugerencias`. Sin él
ese endpoint responde 503 y los botones de la pantalla principal no cargan.

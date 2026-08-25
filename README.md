# TechMind AI — API y artefactos

La API de clasificación de contenido técnico, con el modelo entrenado y la
matriz del histórico, lista para levantar y probar.

Proyecto: [G9-LATAM-Team-46](https://github.com/No-Country-simulation/G9-LATAM-Team-46)

---

## Levantarla

```bash
git clone https://github.com/CbasLugo/artefactos-ml.git
cd artefactos-ml

python preparar.py                       # coloca los artefactos donde van
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Abrir **http://127.0.0.1:8000/docs** — cada endpoint tiene un botón
*Try it out* para probarlo sin escribir código.

Arranca sin base de datos y sin clave de DeepSeek.

---

## Qué probar

**`POST /contenido`** es el principal. Pegar en *Try it out*:

```json
{
  "titulo": "Despliegue con Docker",
  "texto": "Contenedores, Kubernetes y pipelines de CI/CD en AWS con Terraform"
}
```

Devuelve la categoría, la confianza, las palabras clave, las categorías
candidatas y hasta tres documentos del histórico que hablan de lo mismo.

Otros casos que vale la pena mirar:

| Entrada | Qué muestra |
|---|---|
| `"Programar en C++ y C#"` | Los términos técnicos llegan enteros al modelo |
| `"Receta de sopa"` / `"Poner agua a hervir con sal"` | `contenidos_relacionados` vacío: no inventa resultados |
| `titulo: "   "`, `texto: "   "` | Responde `422`, no un error interno |

**`GET /sugerencias`** devuelve los 15 términos de los botones de la
pantalla principal, sin parámetros.

---

## Los artefactos

| Archivo | Peso | Qué es |
|---|--:|---|
| `modelo_techmind_v2.joblib` | 5,9 MB | El clasificador |
| `matriz_historica.pkl` | 50,2 MB | Los 38.257 documentos del corpus vectorizados |
| `sugerencias_botones.json` | 2 KB | Términos de los botones |

Viven en la raíz. `preparar.py` los copia a `backend/app/ml/`, que es donde
la API los busca. No se versiona la copia.

En producción no hace falta ese paso: la API los descarga sola si no los
encuentra, usando estas variables:

```
MODELO_URL=https://github.com/CbasLugo/artefactos-ml/raw/main/modelo_techmind_v2.joblib
MATRIZ_HISTORICA_URL=https://github.com/CbasLugo/artefactos-ml/raw/main/matriz_historica.pkl
SUGERENCIAS_BOTONES_URL=https://github.com/CbasLugo/artefactos-ml/raw/main/sugerencias_botones.json
```

### El clasificador

`Pipeline` de scikit-learn con el vectorizador adentro: recibe texto crudo
y devuelve la predicción, así el preprocesamiento no puede desincronizarse
del modelo.

TF-IDF con unigramas y bigramas sobre 60.000 términos, más Regresión
Logística con `class_weight='balanced'` y `C=4.0`, en 8 categorías.

F1 macro **0,7549** sobre un test de 7.652 textos que conserva la
distribución real. Validación cruzada de 5 particiones: **0,7508 ± 0,0019**.

### La matriz histórica

| Clave | Contenido |
|---|---|
| `vectorizador` | El `TfidfVectorizer` con el que se construyó |
| `matriz` | Matriz dispersa CSR de 38.257 × 20.000 |
| `ids` | Identificador de cada documento |
| `categorias` | Categoría de cada documento |
| `titulos` | Título de cada documento |
| `extractos` | Primeros 200 caracteres del cuerpo |

El vectorizador viaja adentro a propósito: un texto nuevo tiene que
vectorizarse con el mismo vocabulario con el que se construyó la matriz, o
los vectores no serían comparables.

No se guarda la matriz de similitudes entre todos los pares: con 38.257
documentos serían más de mil cuatrocientos millones de valores, casi todos
cercanos a cero. Se guardan los vectores y se compara contra ellos
únicamente el texto de la consulta.

---

## Diferencias con el backend del repositorio principal

Dos cambios, los dos en cómo se arma el contenido relacionado:

**Cada documento trae su extracto.** La matriz ahora guarda los primeros
200 caracteres del cuerpo, así la tarjeta muestra de qué trata y no solo el
título. `recomendador.py` lo lee con `.get()`, de modo que una matriz
anterior sin esa clave sigue cargando y el campo viaja vacío.

**Se prioriza la categoría predicha.** El clasificador y el recomendador
miran el texto en dos espacios vectoriales distintos —60.000 términos
contra 20.000— así que no siempre coinciden: un texto de Jetpack Compose se
clasificaba como Mobile con total seguridad y el vecino más cercano caía en
DevOps / Cloud. Se piden más candidatos y se reordenan poniendo primero los
del mismo tema. No se filtra: si la categoría tiene pocos documentos
parecidos, los del resto siguen estando y la sección no queda vacía.

Medido sobre seis consultas, una por categoría: el primer relacionado pasó
de coincidir 4 veces a coincidir 6.

---

## Versiones

Los artefactos son pickles creados con Python 3.12:

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

Hay que regenerar los tres, no solo el `.joblib`: la matriz guarda su
propio vectorizador y las sugerencias salen del mismo vocabulario. Si el
modelo cambia y ellos no, quedan hablando de vocabularios distintos.

Se generan con los notebooks de `machine_learning/` en el repositorio del
proyecto.

# RAG para Consulta del Sector Energético Colombiano

**Taller Final — Aplicaciones de Machine Learning / NLP**  
**Maestría en Inteligencia Artificial**

**Equipo:** Alejandro Rubiano · Juan Camilo Sanmiguel · Juan Sebastian Londoño

---

## Descripción

Sistema de Recuperación Aumentada por Generación (RAG) diseñado para responder preguntas en lenguaje natural sobre documentos técnicos del sector energético colombiano. El sistema combina búsqueda semántica con embeddings multilingüe y un modelo de lenguaje generativo pequeño para producir respuestas con fuentes citadas.

**Documentos base:**
- Plan Energético Nacional 2050 — UPME (PEN 2050)
- Informe del Sector Minero Energético Colombiano

---

## Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────────────┐
│                     PIPELINE RAG COMPLETO                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ETAPA 1: INDEXACIÓN (offline — se ejecuta una vez)              │
│  ─────────────────────────────────────────────────               │
│                                                                  │
│  [PDFs]                                                          │
│    │  pypdf                                                       │
│    ▼                                                             │
│  [Extracción de texto por página]                                │
│    │  limpieza con regex                                         │
│    ▼                                                             │
│  [Segmentación por grupos de 20 páginas → 7 documentos]          │
│    │  ventana deslizante 200 palabras / overlap 50               │
│    ▼                                                             │
│  [212 chunks con metadatos (doc, página, chunk_id)]              │
│    │  sentence-transformers                                       │
│    │  paraphrase-multilingual-MiniLM-L12-v2                       │
│    ▼                                                             │
│  [Embeddings 384-dim por chunk]                                  │
│    │  L2 normalization                                            │
│    ▼                                                             │
│  [Índice FAISS FlatIP]                                           │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ETAPA 2: RECUPERACIÓN Y GENERACIÓN (online — por consulta)      │
│  ────────────────────────────────────────────────────────        │
│                                                                  │
│  [Pregunta del usuario]                                          │
│    │  sentence-transformers (mismo modelo)                        │
│    ▼                                                             │
│  [Query embedding 384-dim]  ──►  [Índice FAISS]                  │
│                                       │  top-k=3 chunks          │
│                                       ▼                          │
│  [Contexto construido con fuentes anotadas]                       │
│    │                                                             │
│    ▼                                                             │
│  ┌─────────────────────────────────────────┐                     │
│  │  Prompt:                                │                     │
│  │  "Contexto: [chunk1][chunk2][chunk3]    │                     │
│  │   Pregunta: [query]                     │                     │
│  │   Responde basándote solo en el         │                     │
│  │   contexto proporcionado."              │                     │
│  └─────────────────────────────────────────┘                     │
│    │  Qwen2.5-0.5B-Instruct                                       │
│    ▼                                                             │
│  [Respuesta generada + fuentes citadas]                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Repositorio

```
NLP_Taller_RAG/
├── Documents/
│   ├── PEN_2050_UPME.pdf              # Plan Energético Nacional 2050
│   └── Sector_Minero_Energetico.pdf   # Informe Sector Minero-Energético
├── notebooks/
│   └── RAG_Sector_Energetico.ipynb   # Notebook principal con el pipeline completo
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
└── SUGERENCIAS_NOTEBOOK.txt          # Sugerencias de mejora para el notebook
```

---

## Stack Tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| Lectura de PDFs | pypdf | ≥4.0.0 |
| Embeddings | sentence-transformers | ≥2.7.0 |
| Modelo de embeddings | paraphrase-multilingual-MiniLM-L12-v2 | 118M params, 384-dim |
| Búsqueda vectorial | FAISS (faiss-cpu) | ≥1.8.0 |
| Modelo generativo | Qwen/Qwen2.5-0.5B-Instruct | ~500M params |
| Framework de modelos | HuggingFace Transformers | ≥4.40.0 |
| Aceleración GPU | PyTorch + CUDA | ≥2.5.1+cu121 |
| Procesamiento numérico | NumPy, scikit-learn | ≥1.26.0 |
| Visualización | Matplotlib, Seaborn | ≥3.8.0 |

---

## Instalación y Uso

### Requisitos previos

- Python 3.10+
- ~4 GB de RAM
- **GPU (opcional pero recomendado):** NVIDIA con CUDA 11.x o superior. Probado en GTX 1650 Ti (4 GB VRAM). El notebook detecta automáticamente si hay GPU disponible y carga los modelos en `cuda` o `cpu` según corresponda.

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/JuanLondono2/RAG_App.git
cd NLP_Taller_RAG

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash
# Abrir Jupyter
jupyter notebook notebooks/RAG_Sector_Energetico.ipynb
```

> **Nota:** El notebook descarga automáticamente los modelos de HuggingFace en la primera ejecución. Se requiere conexión a internet y ~1.5 GB de espacio en disco.

---

## Proceso de Preprocesamiento

### Carga y Limpieza de PDFs

Los dos documentos PDF se cargan página a página con `pypdf`. Cada página se limpia eliminando espacios múltiples y saltos de línea redundantes usando expresiones regulares.

**Estadísticas del corpus:**
- Total de páginas procesadas: 117
- Promedio de palabras por página: 237.73
- Documentos segmentados: 7

### Segmentación de Documentos

Los PDFs se dividen en segmentos lógicos agrupando páginas de a 20, generando 7 "documentos" manejables. Este enfoque facilita la trazabilidad de las fuentes.

### Chunking

Sobre cada segmento se aplica una ventana deslizante:
- **Tamaño del chunk:** 200 palabras
- **Overlap:** 50 palabras
- **Filtro mínimo:** Se descartan chunks con menos de 40 palabras
- **Chunks resultantes:** 212 con promedio de 155.11 palabras

### Embeddings

Cada chunk se convierte en un vector de 384 dimensiones usando `paraphrase-multilingual-MiniLM-L12-v2`. Los vectores se normalizan con L2 antes de indexarse en FAISS, lo que permite usar el producto interno como aproximación a la similitud coseno.

---

## Resultados y Hallazgos

### Evaluación de Recuperación (Retrieval)

Se diseñaron 5 preguntas de prueba con palabras clave esperadas verificables en el corpus. Las métricas se calculan con chunking semántico por límites de oración (versión actual).

| # | Pregunta (resumida) | Keywords esperadas | Recall@5 | Precision@5 | MRR@5 |
|---|---|---|---|---|---|
| 1 | Importancia del sector minero-energético para la economía | 7%, PIB, 34%, IED, 56%, exportaciones | 1.0 | 0.60 | 1.0 |
| 2 | Porcentaje de exportaciones del sector en 2019 | 56%, exportaciones | 1.0 | 0.40 | 1.0 |
| 3 | Principales sectores de consumo de energía | transporte, industrial, residencial | 1.0 | 0.60 | 1.0 |
| 4 | Objetivo del Plan Energético Nacional 2020-2050 | modelo energético, sostenible, 2050 | 1.0 | 0.40 | 1.0 |
| 5 | Fuentes de energía con mayor participación futura | energías renovables, solar fotovoltaica | 0.0 | 0.40 | 0.0 |

| Métrica | Valor |
|---|---|
| **Recall@5** | **0.80** |
| **Precision@5** | **0.56** |
| **MRR@5** | **0.80** |

> El chunking semántico (por límites de oración) mejoró la coherencia interna de los fragmentos respecto al chunking por palabras original, pero redistribuyó las fronteras de algunos chunks. La pregunta 5 dejó de recuperar la keyword "solar fotovoltaica" en el top-5, lo que redujo el Recall global de 1.0 a 0.80.

El componente de recuperación muestra buen rendimiento general. Los embeddings multilingüe capturan correctamente la semántica de preguntas técnicas en español.

### Evaluación de Generación

El modelo Qwen2.5-0.5B-Instruct genera respuestas coherentes cuando el contexto es claro. Sin embargo, se observó el siguiente comportamiento:

**Respuesta correcta (Q1):**
```
El sector minero energético representa aproximadamente el 7% del PIB colombiano
y el 56% de los ingresos nacionales por exportaciones...
[Fuente: Sector_Minero_Energetico.pdf_parte_0, página 4]
```

**Problema identificado:**
En preguntas donde el contexto recuperado contiene múltiples cifras numéricas de diferentes conceptos (7% PIB, 56% exportaciones, 34% IED), el modelo tiende a mezclar los valores al construir la respuesta, asociando incorrectamente números con conceptos.

### Heatmap de Similaridad Coseno

La visualización de similitud entre las 5 preguntas y los primeros 50 chunks muestra:
- Distribución de relevancia a través de múltiples chunks (no concentrada)
- Scores del chunk más relevante: entre 0.77 y 0.83
- El "gap" entre el chunk top-1 y el resto es moderado (~0.05-0.08), lo que sugiere que el corpus tiene buena densidad semántica relevante

---

## Limitaciones del Sistema

1. **Confusión de cifras numéricas:** El modelo generativo mezcla valores numéricos cuando el contexto contiene varias figuras. Origen probable: chunks que agrupan múltiples estadísticas sin separación semántica clara.

2. **Chunking basado en palabras:** La segmentación por ventana de palabras puede cortar oraciones a la mitad o agrupar conceptos distantes.

3. **Sin validación post-generación:** No existe mecanismo que verifique si la respuesta generada es coherente con las fuentes recuperadas.

4. **Corpus limitado:** Solo dos documentos. El sistema no tiene cobertura sobre aspectos del sector energético no incluidos en ambos PDFs.

5. **Modelo generativo pequeño:** Qwen2.5-0.5B es un modelo de 500M de parámetros. Modelos más grandes producirían respuestas más precisas.

6. **Índice no persistido:** El índice FAISS se reconstruye en cada ejecución (~3-5 minutos).

---

## Mejoras Futuras

| Prioridad | Mejora |
|---|---|
| Alta | Chunking semántico respetando límites de párrafo y oraciones |
| Alta | Persistencia del índice FAISS (guardar/cargar en disco) |
| Media | Usar modelo generativo más capaz (Llama 3.2 3B o API externa) |
| Media | Ampliar métricas de evaluación (Precision@k, MRR, BERTScore) |
| Media | Agregar validación de respuestas contra fuentes (faithfulness) |
| Baja | Expandir corpus con más documentos del sector energético |
| Baja | Explorar modelos de embedding mayores (multilingual-e5-large) |
| Baja | Interfaz de usuario (Gradio o Streamlit) |

---

## Conceptos Clave

**RAG (Retrieval-Augmented Generation):** Paradigma que combina búsqueda de información con generación de texto. En lugar de depender solo de la memoria del modelo de lenguaje, el sistema recupera contexto relevante de una base de conocimiento y lo entrega al modelo junto con la pregunta.

**FAISS (Facebook AI Similarity Search):** Librería para búsqueda eficiente de vecinos más cercanos en espacios de alta dimensionalidad. Permite encontrar los embeddings más similares a una consulta en milisegundos.

**Embeddings multilingüe:** Representaciones vectoriales de texto que capturan significado semántico y funcionan para múltiples idiomas. El modelo `paraphrase-multilingual-MiniLM-L12-v2` fue entrenado en 50+ idiomas.

**Chunking con overlap:** Técnica de segmentación donde cada fragmento comparte palabras con el fragmento anterior, preservando contexto en los límites entre chunks.

---

## Referencias

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- UPME (2019). *Plan Energético Nacional 2020-2050: La transformación energética que habilita el desarrollo sostenible*. Ministerio de Minas y Energía, Colombia.
- Johnson, J. et al. (2017). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.
- Qwen Team (2024). *Qwen2.5 Technical Report*. Alibaba Cloud.

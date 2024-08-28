# Charla HL7 Experience Day Colombia
## Autores

- Andres Rosso
- Daniel Fula

# Análisis de Noticias Médicas con LDA

## Descripción del Proyecto

Este proyecto utiliza el modelo de Alocación Latente de Dirichlet (LDA) para analizar y categorizar noticias médicas recolectadas durante un periodo de tres años. El objetivo es identificar los principales temas discutidos y analizar cómo varían estos temas a lo largo del tiempo.


## Requisitos

Para ejecutar el código de este proyecto, necesitarás Python 3.x y las siguientes bibliotecas:

- `gensim` para la modelización de LDA
- `pandas` para la manipulación de datos
- `matplotlib` para la visualización de datos

Puedes instalar todas las dependencias necesarias con el siguiente comando:

```bash
pip install gensim pandas matplotlib
```

## Preparación de Datos

Los datos consisten en un conjunto de noticias médicas recolectadas durante tres años. El siguiente código carga los datos, los preprocesa y los prepara para el modelado LDA:

```python
import pandas as pd

# Carga de datos
data = pd.read_csv('path_to_data.csv')

# Preprocesamiento de texto
def preprocess_text(text):
    # Aquí irían las operaciones de preprocesamiento como la tokenización,
    # eliminación de stopwords, etc.
    return processed_text

data['processed_text'] = data['news_text'].apply(preprocess_text)
```

## Modelado LDA

Utilizamos la biblioteca `gensim` para crear y entrenar el modelo LDA. El siguiente código configura y entrena el modelo LDA:

```python
from gensim import corpora, models

# Preparación de datos para LDA
dictionary = corpora.Dictionary(data['processed_text'])
corpus = [dictionary.doc2bow(text) for text in data['processed_text']]

# Creación y entrenamiento del modelo LDA
lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

# Guardar el modelo
lda_model.save('lda_model.model')
```

## Análisis de Resultados

El análisis se realiza calculando la distribución de temas por mes y visualizando estos datos:

```python
import matplotlib.pyplot as plt

# Cálculo de distribución de temas por mes
topic_distribution = calculate_topic_distribution(lda_model, corpus)

# Visualización
plt.figure(figsize=(10,5))
plt.plot(topic_distribution['date'], topic_distribution['topic_1'])
plt.title('Distribución del Tema 1 a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('Distribución del tema')
plt.show()
```

## Conclusión

Este proyecto permite obtener una comprensión detallada de los temas prevalentes en las noticias médicas y cómo estos temas cambian a lo largo del tiempo, lo cual es crucial para el seguimiento de tendencias en el sector salud.
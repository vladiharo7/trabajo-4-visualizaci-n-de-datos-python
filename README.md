# Trabajo 4: Visualización de datos con Matplotlib y Seaborn

## Descripcion

Creación de visualizaciones con Matplotlib y Seaborn para analizar un conjunto de datos de ventas minoristas.

## Contexto

Si usas Visual Studio Code en la plataforma, al abrirlo te aparecerá el dataset superstore_dataset2012.csv, si optas por usar tu propio entorno en tu ordenador y subir la solución en zip o github entonces puedes descargar manualmente el dataset.

## Requisitos

* Crear al menos un gráfico univariante (histograma, diagrama de barras o diagrama de caja) utilizando Matplotlib
* Crear al menos un gráfico univariante utilizando Seaborn
* Implementar al menos un gráfico bivariante (dispersión, líneas o barras agrupadas) con Matplotlib
* Implementar al menos un gráfico bivariante con Seaborn
* Crear una visualización multivariante (como un gráfico de pares o un heatmap de correlación) con Seaborn
* Personalizar los gráficos (títulos, etiquetas de ejes, paletas de colores)
* Organizar múltiples visualizaciones en una figura usando subplots
* Guardar al menos una figura generada como archivo de imagen
* Incluir comentarios explicativos sobre las conclusiones obtenidas de cada visualización
* Utilizar el dataset superstore_dataset2012.csv proporcionado

## Instrucciones

### Configura tu entorno de trabajo
Crea un nuevo archivo Python (.py) o un notebook Jupyter (.ipynb). Importa las bibliotecas necesarias (pandas, matplotlib, seaborn) y carga el dataset superstore_dataset2012.csv que ya está disponible en tu entorno.

### Explora y prepara los datos
Realiza una exploración inicial del dataset para entender su estructura. Verifica los tipos de datos, valores nulos y realiza las transformaciones necesarias (como convertir fechas al formato adecuado).

### Crea visualizaciones univariantes con Matplotlib
Implementa al menos un histograma o diagrama de barras utilizando Matplotlib para visualizar la distribución de una variable numérica (como Ventas o Beneficios) o la frecuencia de una variable categórica (como Categoría o Segmento).

### Crea visualizaciones univariantes con Seaborn
Utiliza Seaborn para crear al menos un diagrama de caja (boxplot) o un gráfico de violín (violinplot) para visualizar la distribución de una variable numérica, posiblemente agrupada por una variable categórica.

### Implementa gráficos bivariantes con Matplotlib
Crea un gráfico de dispersión o de líneas con Matplotlib para mostrar la relación entre dos variables numéricas (por ejemplo, Ventas vs. Beneficios) o la evolución temporal de una variable.

### Implementa gráficos bivariantes con Seaborn
Utiliza Seaborn para crear un gráfico bivariante como un gráfico de barras agrupadas, un gráfico de dispersión con regresión (regplot) o un gráfico de líneas mejorado.

### Crea una visualización multivariante con Seaborn
Implementa un heatmap de correlación o un pairplot para visualizar las relaciones entre múltiples variables numéricas del dataset.

### Organiza visualizaciones en subplots
Crea una figura con múltiples subplots que muestre al menos 4 visualizaciones diferentes, organizadas de manera coherente y con un título general.

### Personaliza las visualizaciones
Mejora la apariencia de tus gráficos añadiendo títulos descriptivos, etiquetas de ejes claras, leyendas cuando sea necesario y utilizando paletas de colores apropiadas.

### Guarda y documenta tu trabajo
Guarda al menos una de tus visualizaciones como archivo de imagen. Añade comentarios a tu código explicando las conclusiones que se pueden extraer de cada visualización y cómo contribuyen al análisis general de los datos.

## Criterios de Calificacion

### Implementación de visualizaciones con Matplotlib
Correcta creación de gráficos univariantes, bivariantes y multivariantes con Matplotlib.

Peso: 40%

### Implementación de visualizaciones con Seaborn

Implementación efectiva de visualizaciones univariantes, bivariantes, multivariantes con Seaborn

Peso: 40%


### Preparación y manejo de datos con Pandas
Correcta carga y preparación del dataset, incluyendo transformaciones necesarias como conversión de fechas, y manejo adecuado de los datos para las visualizaciones.

Peso: 20%

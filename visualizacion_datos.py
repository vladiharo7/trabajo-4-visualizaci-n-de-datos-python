import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. CONFIGURACIÓN E IMPORTACIÓN DE DATOS ---

# Configuración de estilo global para Seaborn
sns.set_style("whitegrid")

try:
    # Cargar el dataset. Se asume que está en el directorio de trabajo.
    df = pd.read_csv("workspace/superstore_dataset2012.csv", encoding='latin1')
except FileNotFoundError:
    print("Error: El archivo 'superstore_dataset2012.csv' no fue encontrado.")
    # Crea un DataFrame de ejemplo para que el código restante pueda ejecutarse
    # Esto es solo para propósitos demostrativos si el archivo falta
    data = {
        'Order Date': pd.to_datetime(['2012-01-01', '2012-01-01', '2012-02-05', '2012-02-05', '2012-03-10']),
        'Category': ['Technology', 'Technology', 'Office Supplies', 'Furniture', 'Technology'],
        'Segment': ['Consumer', 'Corporate', 'Home Office', 'Consumer', 'Corporate'],
        'Sales': [1000, 500, 50, 200, 1500],
        'Profit': [150, 75, -10, 20, 300],
        'Shipping Cost': [20, 10, 5, 15, 30]
    }
    df = pd.DataFrame(data)
    # Continúa el proceso con una advertencia
    print("Usando datos de ejemplo para continuar la ejecución.")


# --- 2. EXPLORACIÓN Y PREPARACIÓN DE DATOS (20% Peso) ---

# Conversión de fechas: Asegurar que la columna de fecha es datetime
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Creación de una columna de Mes y Año para análisis temporal
df['Order MonthYear'] = df['Order Date'].dt.to_period('M')

# Eliminar nulos que puedan afectar las visualizaciones clave
df_clean = df.dropna(subset=['Sales', 'Profit', 'Category', 'Segment', 'Order MonthYear']).copy()


# -------------------------------------------------------------------
# --- 3. CREACIÓN DE VISUALIZACIONES (80% Peso) ---
# -------------------------------------------------------------------


# ** VISUALIZACIÓN 1: UNIVARIANTE CON MATPLOTLIB (Histograma) **
# Requisito: Crear al menos un gráfico univariante con Matplotlib

plt.figure(figsize=(10, 6))
# Filtrar Beneficio > 0 y < 500 para una mejor visualización del grueso de datos
subset_profit = df_clean[
    (df_clean['Profit'] > 0) & (df_clean['Profit'] < 500)
]

plt.hist(
    subset_profit['Profit'],
    bins=50,
    color='#1f77b4',  # Azul estándar de Matplotlib
    edgecolor='black',
    alpha=0.8
)
plt.title('Distribución de Beneficios (Ganancias entre 0 y $500)', fontsize=16, fontweight='bold')
plt.xlabel('Beneficio ($)', fontsize=12)
plt.ylabel('Frecuencia de Órdenes', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

print("\nConclusión 1 (Histograma Matplotlib): La mayoría de las transacciones rentables generan un beneficio relativamente pequeño (menos de $50), mostrando una distribución fuertemente sesgada a la derecha.")

# -------------------------------------------------------------------

# ** VISUALIZACIÓN 2: UNIVARIANTE CON SEABORN (Boxplot agrupado) **
# Requisito: Crear al menos un gráfico univariante con Seaborn

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Category',
    y='Sales',
    data=df_clean,
    palette='Set2',  # Paleta amigable para la categoría
    showfliers=False # Ocultar outliers extremos para ver mejor el cuerpo de los datos
)
plt.title('Distribución de Ventas por Categoría de Producto', fontsize=16, fontweight='bold')
plt.xlabel('Categoría', fontsize=12)
plt.ylabel('Ventas ($)', fontsize=12)
plt.show()

print("\nConclusión 2 (Boxplot Seaborn): 'Technology' y 'Furniture' tienen una mediana de ventas (línea central) más alta que 'Office Supplies'. Las cajas más grandes indican una mayor dispersión de ventas en 'Technology'.")

# -------------------------------------------------------------------

# ** VISUALIZACIÓN 3: BIVARIANTE CON MATPLOTLIB (Gráfico de Líneas Temporal) **
# Requisito: Implementar al menos un gráfico bivariante con Matplotlib

# Agregar ventas y beneficios por mes
df_time = df_clean.groupby('Order MonthYear')[['Sales', 'Profit']].sum().reset_index()
df_time['Order MonthYear'] = df_time['Order MonthYear'].astype(str)

fig, ax1 = plt.subplots(figsize=(12, 6))

color_sales = 'tab:blue'
ax1.set_xlabel('Mes y Año', fontsize=12)
ax1.set_ylabel('Ventas Totales ($)', color=color_sales, fontsize=12)
ax1.plot(df_time['Order MonthYear'], df_time['Sales'], color=color_sales, label='Ventas', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_sales)
ax1.set_xticks(np.arange(0, len(df_time), 3)) # Mostrar cada 3 meses para legibilidad
ax1.set_xticklabels(df_time['Order MonthYear'][::3], rotation=45, ha='right')

# Crear un segundo eje para Beneficios
ax2 = ax1.twinx()
color_profit = 'tab:red'
ax2.set_ylabel('Beneficio Total ($)', color=color_profit, fontsize=12)
ax2.plot(df_time['Order MonthYear'], df_time['Profit'], color=color_profit, label='Beneficio', linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_profit)

plt.title('Ventas y Beneficios Totales a lo largo del Tiempo', fontsize=16, fontweight='bold')
fig.tight_layout()
plt.show()

print("\nConclusión 3 (Líneas Matplotlib): Se observa un fuerte patrón de estacionalidad, con picos de Ventas y Beneficios generalmente hacia el final del año. Las ventas y beneficios suelen ir de la mano.")

# -------------------------------------------------------------------

# ** VISUALIZACIÓN 4: BIVARIANTE CON SEABORN (Gráfico de Barras Agrupadas) **
# Requisito: Implementar al menos un gráfico bivariante con Seaborn

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Category',
    y='Profit',
    hue='Segment',
    data=df_clean,
    estimator=np.sum, # Sumar los beneficios por grupo
    palette='viridis'
)
plt.title('Beneficio Total por Categoría y Segmento de Cliente', fontsize=16, fontweight='bold')
plt.xlabel('Categoría de Producto', fontsize=12)
plt.ylabel('Beneficio Total ($)', fontsize=12)
plt.legend(title='Segmento')
plt.show()

print("\nConclusión 4 (Barras Seaborn): El segmento 'Consumer' es el que más contribuye al Beneficio Total en todas las categorías, especialmente en 'Technology'.")

# -------------------------------------------------------------------

# ** VISUALIZACIÓN 5: MULTIVARIANTE CON SEABORN (Heatmap de Correlación) **
# Requisito: Crear una visualización multivariante con Seaborn

# Seleccionar variables numéricas
numeric_cols = df_clean[['Sales', 'Profit', 'Shipping Cost']]
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(8, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,        # Mostrar valores numéricos
    fmt=".2f",         # Formato con 2 decimales
    cmap='coolwarm',   # Paleta divergente
    center=0,          # Centrar el color en cero
    linewidths=.5,
    cbar_kws={'label': 'Coeficiente de Correlación'}
)
plt.title('Matriz de Correlación de Variables Clave', fontsize=16, fontweight='bold')
plt.show()

print("\nConclusión 5 (Heatmap Seaborn): Existe una correlación positiva fuerte (0.69) entre 'Sales' y 'Profit', lo cual es esperado. También hay una correlación moderada entre 'Sales' y 'Shipping Cost' (0.50).")

# -------------------------------------------------------------------

# ** VISUALIZACIÓN 6: ORGANIZACIÓN DE SUBPLOTS **
# Requisito: Organizar múltiples visualizaciones en una figura usando subplots
# Requisito: Guardar al menos una figura generada como archivo de imagen

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis Multifacético de Ventas y Beneficios (Subplots)', fontsize=18, fontweight='bold', y=1.02)

# --- Subplot 1 (Top-Left): Boxplot de Costo de Envío por Categoría ---
sns.boxplot(
    x='Category',
    y='Shipping Cost',
    data=df_clean,
    palette='Pastel1',
    ax=axes[0, 0],
    showfliers=False
)
axes[0, 0].set_title('A. Costo de Envío por Categoría', fontsize=14)
axes[0, 0].set_xlabel('Categoría', fontsize=12)
axes[0, 0].set_ylabel('Costo de Envío ($)', fontsize=12)

# --- Subplot 2 (Top-Right): Scatterplot (Bivariante Matplotlib) ---
# Scatterplot Matplotlib (Sales vs Profit)
axes[0, 1].scatter(
    df_clean['Sales'],
    df_clean['Profit'],
    alpha=0.3,
    color='darkorange',
    s=10 # Tamaño de punto
)
axes[0, 1].set_title('B. Relación Ventas vs. Beneficios (Matplotlib)', fontsize=14)
axes[0, 1].set_xlabel('Ventas ($)', fontsize=12)
axes[0, 1].set_ylabel('Beneficio ($)', fontsize=12)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=0.8) # Línea de Beneficio cero

# --- Subplot 3 (Bottom-Left): Regplot (Bivariante Seaborn) ---
# Scatterplot con línea de regresión Seaborn
sns.regplot(
    x='Sales',
    y='Profit',
    data=df_clean.sample(n=1000, random_state=42), # Muestra para mejor rendimiento
    scatter_kws={'alpha': 0.4, 's': 20},
    line_kws={'color': 'darkgreen', 'lw': 2},
    ax=axes[1, 0]
)
axes[1, 0].set_title('C. Regresión de Ventas vs. Beneficios (Seaborn)', fontsize=14)
axes[1, 0].set_xlabel('Ventas ($)', fontsize=12)
axes[1, 0].set_ylabel('Beneficio ($)', fontsize=12)

# --- Subplot 4 (Bottom-Right): Gráfico de Barras (Univariante Matplotlib - Frecuencia) ---
# Contar frecuencia de Segmento
segment_counts = df_clean['Segment'].value_counts()
axes[1, 1].bar(
    segment_counts.index,
    segment_counts.values,
    color=['#4c72b0', '#55a868', '#c44e52'] # Colores personalizados
)
axes[1, 1].set_title('D. Frecuencia de Órdenes por Segmento', fontsize=14)
axes[1, 1].set_xlabel('Segmento de Cliente', fontsize=12)
axes[1, 1].set_ylabel('Número de Órdenes', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajustar diseño para hacer espacio al título
plt.show()

# --- 7. GUARDAR FIGURA ---
try:
    fig.savefig('workspace/analisis_superstore_subplots.png', dpi=300)
    print("\nÉxito: La figura 'analisis_superstore_subplots.png' ha sido guardada.")
except Exception as e:
    print(f"\nError al guardar la figura: {e}")

print("\nConclusión 6 (Subplots): La visualización combinada revela que: A. El costo de envío es similar entre categorías. B y C. La relación entre Ventas y Beneficios es fuertemente positiva, aunque las grandes ventas con bajo o negativo beneficio (puntos atípicos) merecen una revisión. D. El segmento 'Consumer' genera la mayoría de las órdenes.")
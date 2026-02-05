# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(
    r"C:\Users\pro pc\Desktop\python1\AI\excel\excel acctualisados\TheBostonHousingDataset.xlsx")

# Convertidor de columnas a listas
CRIM = df['CRIM'].tolist()
ZN = df['ZN'].tolist()
INDUS = df['INDUS'].tolist()
CHAS = df['CHAS'].tolist()
NOX = df['NOX'].tolist()
RM = df['RM'].tolist()
AGE = df['AGE'].tolist()
DIS = df['DIS'].tolist()
RAD = df['RAD'].tolist()
TAX = df['TAX'].tolist()
PTRATIO = df['PTRATIO'].tolist()
B = df['B'].tolist()
LSTAT = df['LSTAT'].tolist()
MEDV = df['MEDV'].tolist()

m_correlacion = df.corr()
print(m_correlacion)

plt.figure(figsize=(10,8))
sns.heatmap(m_correlacion, annot=True, cmap='crest', fmt='.2f')
plt.title('CORRELATION MATRIX')
plt.show()

caracteristicas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Plots agrupados
plt.figure(figsize=(15,12))
for i, caracteristica in enumerate(caracteristicas):
    plt.subplot(4,4,i+1)
    plt.scatter(df[caracteristica], df['MEDV'], alpha=0.9)
    plt.title(f'{caracteristica} vs MEDV')
    plt.xlabel(caracteristica)
    plt.ylabel('MEDV')
plt.tight_layout()
plt.show()


# Estadistica descriptiva

print(df.describe())
# Visualización de Datos
print('\nVisualización de Datos...')
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df_scaled['TAX'], df_scaled['MEDV'], color='blue')
plt.title('Precio vs Impuestos (Dataset Procesado)')
plt.xlabel('Impuestos (Dataset Procesado)')
plt.ylabel('Precio ($)')

plt.subplot(1, 2, 2)
plt.scatter(df_scalado['RM'], df_scalado['MEDV'], color='green')
plt.title('Precio vs Cantidad de Habitaciones (Dataset Procesado)')
plt.xlabel('Habitaciones (Dataset Procesado)')
plt.ylabel('Precio ($)')

plt.tight_layout()
plt.show()
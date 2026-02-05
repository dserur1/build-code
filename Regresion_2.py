# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_excel(r'C:\Users\pro pc\Desktop\python1\AI\excel\data_regresion2.xlsx')
print(df.head())

#listas de las columnas nobres
tamano = df["Tamaño (m2)"].tolist()
habitaciones = df["Habitaciones"].tolist()
precio = df["Precio"].tolist()

x = np.array(list (zip(tamano,habitaciones))) 
y = np.array(precio)
print("\nDividiendo los datos en conjuntos de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"tamano enrenamiento x:{X_train.shape}")
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(tamano , precio, color='blue')
plt.title('Precio vs Tamaño de la Casa')
plt.xlabel('Tamaño (m²)')
plt.ylabel('Precio ($)')
plt.subplot(1,2,2)
plt.scatter(habitaciones , precio, color='red')
plt.title('Precio vs habitaciones de la Casa')
plt.xlabel('habitaciones (m²)')
plt.ylabel('Precio ($)')
plt.show()
modelo = LinearRegression()
modelo.fit(X_train,y_train)
y_predict = modelo.predict(X_test)  
print(y_predict)
error = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print("\nMétricas de Evaluación:")
print(f"Error cuadrático medio (error): {error:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.4f}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predict, color='blue', label="Predicciones")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label="Línea Ideal")
plt.title('Predicción vs Valores Reales')
plt.xlabel('Valores Reales ($)')
plt.ylabel('Predicciones ($)')
plt.legend()
plt.grid(True)
plt.show()
# PREDICION CON NUEVOS DATOS.
print("ingresa parametros para calcular valor de un precio")
area = float (input("input your area: " ))
habitacion = int (input("input your rooms: "))
xpredict = np.array([[area,habitacion]])
print(xpredict)
y_predict = modelo.predict(xpredict)
print(f"este es tu prediccion de presio: {y_predict[0]} ")
r2 = r2_score(y_test, y_predict)
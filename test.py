# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
#from tensorflow.python.keras.layers import Dense, Input 
#from tensorflow.python.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ************************************
# Leer datos desde el archivo Excel
print('\nCargando datos...')
df = pd.read_excel(r'C:\Users\pro pc\Desktop\python1\AI\excel\data_clasificacion_Insectos.xlsx')

# ************************************
# Mostrar las primeras filas del DataFrame para verificar que se ha leído correctamente
#print(df.head())


# Preprocesamiento de los datos
print('\nPreprocesando los datos...')


# ************************************
# Convertir atributos categóricos a numéricos
print('\nConvertir atributos categóricos a numéricos...')
label_encoder = LabelEncoder()
df['Color_Principal'] = label_encoder.fit_transform(df['Color_Principal'])
df['Tipo_Alas'] = label_encoder.fit_transform(df['Tipo_Alas'])
df['Habitat'] = label_encoder.fit_transform(df['Habitat'])


# ************************************
# características (atributos) y las etiquetas (clase) del DataFrame
print('\nSeparar características y etiquetas...')
X = df[['Tamano_cm', 'Numero_Patas', 'Color_Principal', 'Tipo_Alas', 'Habitat']].values
y = df['Etiqueta'].values


# ************************************
# Escalar las características
print('\nEscalar las características...')
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ************************************
# Codificar etiquetas
# Se codifican las etiquetas para que sean valores numéricos.
print('\nCodificar etiquetas...')
y = label_encoder.fit_transform(y)


# ************************************
# Dividir en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
print('\nDividir Dataset de entrenamiento y prueba...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTamaño de X_train:", X_train.shape)
print("Tamaño de X_test:", X_test.shape)
print("Tamaño de y_train:", y_train.shape)
print("Tamaño de y_test:", y_test.shape,"\n\n")


# ************************************
# Construcción del modelo de red neuronal
print('\nConstrucción del modelo de red neuronal...')
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Capa oculta 1
model.add(Dense(32, activation='relu'))  # Capa oculta 2
model.add(Dense(16, activation='relu'))  # Capa oculta 3
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Capa de salida

# Compilar el modelo
print('\nCompilando el modelo...')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo usando retropropagación
print('\nEntrenando el modelo usando retropropagación...')
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))
print('Fase de entrenamiento finalizada')


# ************************************
# Guardar el modelo completo
# Formato de archivo portable HDF5 (Hierarchical Data Format version 5)
print('\nGuardando el modelo...')
model.save('modelo_RNA_BP.h5')

# ************************************
# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print("\nPrecisión del modelo en el conjunto de prueba: {:.2f}%".format(accuracy * 100))


# ************************************
# Visualización de los Resultados

# Gráfico de precisión
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Gráfico de pérdida
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()



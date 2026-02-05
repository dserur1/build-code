
#Acer IA que me de nombres de incesctos atravez de informacion de dataframe que tenga modelo de entrenamiento que sea knn, que evalue para cada k que aroje menos herror,que se evalue el modelo que se realize predicciones con datos de prueba, que se grafique matris de confuccion y que grafique para cada knn que se aga.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd                                             
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

#este es el data farame
df = pd.read_excel(r"C:\Users\pro pc\Desktop\python1\AI\excel\data_clasificacion_Insectos.xlsx")
#este printea los datos del dataframe
print(df.head())

# transformar datos cualitativos a cauantitativos!

# Crear vectorizadores para atributos de texto
vectorizer_color = TfidfVectorizer()
vectorizer_tipo_alas = TfidfVectorizer()
vectorizer_habitat = TfidfVectorizer()

# Crear escalador para atributos numéricos
scaler = MinMaxScaler()

# Transformar atributos de texto en matrices dispersas
X_color = vectorizer_color.fit_transform(df['Color_Principal'])
X_tipo_alas = vectorizer_tipo_alas.fit_transform(df['Tipo_Alas'])
X_habitat = vectorizer_habitat.fit_transform(df['Habitat'])

# Transformar atributos numéricos en matrices escaladas
X_tamano = scaler.fit_transform(df[['Tamano_cm']])
X_numero_patas = scaler.fit_transform(df[['Numero_Patas']])

# Combinar todas las matrices en una sola matriz dispersa
X_combined = hstack([X_tamano, X_numero_patas, X_color, X_tipo_alas, X_habitat])

# Mostrar la forma de la matriz combinada
print(f"Forma de la matriz combinada: {X_combined.shape}")

#garfica de distribucion de las clasess
df['Etiqueta'].value_counts().plot(kind='bar', title='Distribución de las Clases')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.show()

# Etiquetas
y = df['Etiqueta']

print('Dividiendo el conjunto de datos en entrenamiento y prueba ...')
#  
X_train , X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)
#Modelo KNN :
Modelo = KNeighborsClassifier (n_neighbors=4,metric='euclidean')
#este es el enrenamiento del modelo
Modelo.fit(X_train, y_train)

# Evaluar el modelo
print('Evaluando el modelo ...\n')
score = Modelo.score(X_test, y_test)
print(f"Precisión del modelo: {score:.2f}")



# Realizar predicciones en los datos de prueba
y_predict = Modelo.predict(X_test)
print(X_test,y_predict)

# Calcular la precisión del modelo
presicion = accuracy_score(y_test,y_predict)
print(f"la precision del model es : {presicion}")

# garfica de matriz de confucion
# Matriz de confusión
cm = confusion_matrix(y_test, y_predict)
print("Matriz de confusión:\n", cm)

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels = np.unique(y)  , yticklabels = np.unique (y) )
plt.xlabel(' PREDICCION ')
plt.ylabel('VERDADERA')
plt.title('Matriz de Confusión')
plt.show() 

Valores_de_k = range(1,21)
Valores = []
for i in Valores_de_k:
    knn = KNeighborsClassifier(n_neighbors = i )
    knn.fit(X_train,y_train)
    presicion = accuracy_score(y_test,y_predict) 
    Valores.append( presicion )
plt.figure(figsize = (10,6) )
plt.plot( Valores_de_k, Valores, marker = 'o') 
plt.ylabel('datods_de_herrores')
plt.xlabel('datos_de_knn')
plt.title('Curva de validacion para diferentes valores de cambio')
plt.show()  





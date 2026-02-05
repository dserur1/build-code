import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

lector = pd.read_excel(r'C:\Users\pro pc\Desktop\python1\AI\excel\data_clasificacion_Insectos.xlsx')
print(lector.head())

vector_color = TfidfVectorizer()
vector_tipo_alas = TfidfVectorizer()
vector_habitat = TfidfVectorizer()
escala = MinMaxScaler()

datos_color = vector_color.fit_transform(lector['Color_Principal'])
datos_tipo_alas = vector_tipo_alas.fit_transform(lector['Tipo_Alas'])
datos_tipo_habitad = vector_habitat.fit_transform(lector['Habitat'])
datos_tamano = escala.fit_transform(lector[['Tamano_cm']])
datos_numero_patas = escala.fit_transform(lector[['Numero_Patas']])

combinacion = hstack([datos_numero_patas,datos_tamano,datos_tipo_habitad,datos_tipo_alas,datos_color])
print(combinacion)

y = lector['Etiqueta']
entrenamiento_x,test_x,entrenamiento_y,test_y = train_test_split(combinacion,y,test_size=0.2, random_state=42)
modelo = MultinomialNB()
modelo.fit(entrenamiento_x,entrenamiento_y)
y_predict = modelo.predict(test_x)
print(y_predict)
exactitud = accuracy_score(test_y,y_predict)
print(f'La exactitud es : {exactitud}')

mc = confusion_matrix(test_y,y_predict)
plt.figure(figsize=(32, 24))
sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()
nombres = y_predict
plt.figure(figsize=(32, 24))
sns.heatmap(mc, annot=True, fmt='d', cmap='Reds', xticklabels= nombres , yticklabels= nombres)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()
puntaje = cross_val_score(modelo,combinacion,y , cv=2 )
print(f'el puntaje es : {puntaje}')
print( "el puntaje medio es :",puntaje.mean())
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#Leyendo el data Frame
df = pd.read_excel(r'C:\Users\pro pc\Desktop\python1\AI\excel\data_clasificacion_Insectos.xlsx')
print(df.head())

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

# Etiquetas
y = df['Etiqueta']

print('Dividiendo el conjunto de datos en entrenamiento y prueba ...')
# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

print('Entrenando el modelo ...')
# Entrenar el árbol de decisión
clf = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=42)
clf.fit(X_train, y_train)


# Evaluar el modelo
print('Evaluando el modelo ...\n')
score = clf.score(X_test, y_test)
print(f"Precisión del modelo: {score:.2f}")

# Validación cruzada
print('\nRealizando validación cruzada ...')
cv_scores = cross_val_score(clf, X_combined, y, cv=3)  # 3-fold cross-validation
print(f"Puntuaciones de la validación cruzada: {cv_scores}")
print(f"Precisión media de la validación cruzada: {cv_scores.mean():.2f}")

# Visualizar el árbol de decisión
print('\nRepresentación del árbol de decisión ...')
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=['Tamano_cm', 'Numero_Patas'] + \
           vectorizer_color.get_feature_names_out().tolist() + \
           vectorizer_tipo_alas.get_feature_names_out().tolist() + \
           vectorizer_habitat.get_feature_names_out().tolist(),
           class_names=np.unique(y).astype(str), rounded=True, proportion=True)
plt.show()

# Realizar predicciones en los datos de prueba
print('\nPredicciones con los datos de prueba ...')
y_pred = clf.predict(X_test)

print(y_pred)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("\nPrecisión:", accuracy)


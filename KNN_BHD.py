import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


# Función para graficar pares de histogramas
def plot_histograms_comparison(df_original, df_scaled):
    for feature in df_original.columns:
        # Crear una nueva figura para cada par de histogramas
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        
        # Histograma de la característica original
        axes[0].hist(df_original[feature], bins=10, edgecolor='black')
        axes[0].set_title(f'Original - {feature}')
        
        # Histograma de la característ
# Histograma de la característica escalada
        axes[1].hist(df_scaled[feature], bins=10, edgecolor='black')
        axes[1].set_title(f'Procesado - {feature}')
        
        plt.tight_layout()
        plt.show()
# Función para detectar y manejar valores atípicos 
# usando el rango intercuartílico (IQR)
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
# funcion mena imputer faltantes por la media
def mean_imputer(df): 
    imputer = SimpleImputer(strategy = 'mean' )
    df_imputado = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)
    return(df_imputado)
def knn_imputer(df):
    imputer = KNNImputer(n_neighbors=5)
    df_imputado = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)
    return(df_imputado)
def feture_scaler(df_clean): 
    scaler = StandardScaler()
    sacled_fetures = scaler.fit_transform(df_clean.drop('MEDV',axis=1))
    
    #qureamos un data frame con caracteristicas escaladas.
    df_scaled = pd.DataFrame(sacled_fetures,columns=df_clean.columns[:-1])
    df_scaled['MEDV'] = df_clean['MEDV'].values
    return(df_scaled)

def feture_log(df_clean):
    # Agremgamos una constate para evitrar logaritmos de cero
    df_log_scaled = df_clean.drop(columns = ['MEDV'].apply(lambda x : np.log1p(x)))
    df_log_scaled['MEDV'] = df_clean['MEDV']
    return(df_log_scaled)


#Codigo base
df = pd.read_excel(
    r"C:\Users\pro pc\Desktop\python1\AI\excel\excel acctualisados\TheBostonHousingDataset.xlsx")
opcion = input("Son dos opciones escrive 1 para imputar elementos faltantes con la media y escrive 2 para imputar elemntos faltantes usando KNN:")
match opcion : 
    case '1':
        df_imputado = mean_imputer(df)
    case '2':
        df_imputado = knn_imputer(df)
    case _ :
        df_imputado = mean_imputer(df) 

#elimina flias duplicadas
df_clean = df_imputado.drop_duplicates()
#elimina valores atipicos
df_clean = remove_outliers_iqr(df_clean)     

opcion = input("son dos opciones: escrive 1 par normalizar o escrive 2 para escalar!")
match opcion:
    case '1':
        df_escalar = feture_log(df_clean)
    case '2':
        df_escalar = feture_scaler(df_clean)
    case _ :
        df_escalar = feture_scaler(df_clean)

print(df_escalar.head())

#Variables de la IA
x = df_escalar.drop(columns=['MEDV'] )
y = df_escalar['MEDV'] 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
Modelo = KNeighborsRegressor(n_neighbors=4)
Modelo.fit(X_train,y_train)   
y_predict = Modelo.predict(X_test)
print(y_predict)
print(f"este es tu prediccion de presio: {round(y_predict[0],2)}")
r2 = r2_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict) 
print(f'este es el herror.. {round(error,2)}, esta es tu preciocion.. {round(r2,2)}') 

#grafica de disparcion de predicciones vs valores reales

plt.figure(figsize=(10,6))


plt.scatter(y_test,y_predict, alpha=0.7, color ='g')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'--r',lw=2)

plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Predichos vs Valores Reales')

plt.show() 

#grafica de residuos
residuo = y_test - y_predict 
plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuo,alpha = 0.7, color = 'b')
plt.axhline(y=0,color= 'y',linestyle = '-' )
plt.ylabel('residuos')
plt.xlabel('valores prebichos mamaguebo')
plt.title('grafica de residuos')
plt.show()

#Curva de validacion para diferentes valores de cambio

Valores_de_k = range(1,21)
Valores = []
for i in Valores_de_k:
    knn = KNeighborsRegressor(n_neighbors = i )
    knn.fit(X_train,y_train)
    y_predict = knn.predict(X_test)
    Valores.append(mean_squared_error(y_test,y_predict))
plt.figure(figsize = (10,6) )
plt.plot( Valores_de_k, Valores, marker = 'o') 
plt.ylabel('datods_de_herrores')
plt.xlabel('datos_de_knn')
plt.title('Curva de validacion para diferentes valores de cambio')
plt.show()  

# grafica de importancia de caracteristicas.
importancia = permutation_importance(Modelo,X_test,y_test)
indicens_ordenados = importancia.importances_mean.argsort()
plt.figure(figsize=(10,6))
plt.barh(x.columns[indicens_ordenados],importancia.importances_mean[indicens_ordenados])
plt.xlabel('Importancia de la Característica')
plt.title('Importancia de las Características')
plt.show()
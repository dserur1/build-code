#librerias
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping,Callback
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Funciones
def mean_imputer(df): 
    imputer = SimpleImputer(strategy = 'mean' )
    df_imputado = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)
    return(df_imputado)
def knn_imputer(df):
    imputer = KNNImputer(n_neighbors=5)
    df_imputado = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)
    return(df_imputado)
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
def feture_log(df_clean):
    # Agremgamos una constate para evitrar logaritmos de cero
    df_log_scaled = df_clean.drop(columns = ['MEDV'].apply(lambda x : np.log1p(x)))
    df_log_scaled['MEDV'] = df_clean['MEDV']
    return df_log_scaled
#qureamos un data frame con caracteristicas escaladas.
def feture_scaler(df_clean): 
    scaler = StandardScaler()
    sacled_fetures = scaler.fit_transform(df_clean.drop('MEDV',axis=1))
    df_scaled = pd.DataFrame(sacled_fetures,columns=df_clean.columns[:-1])
    df_scaled['MEDV'] = df_clean['MEDV'].values
    return(df_scaled)
#clase calcula R2 epocas (error y precision)
class R2Callback(Callback):
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.train_r2 = []
        self.val_r2 = []

    def on_epoch_end(self, epoch, logs=None):
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        self.train_r2.append(train_r2)
        self.val_r2.append(val_r2)
        print(f'Epoch {epoch+1}: Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}')



#codigo Base
df = pd.read_excel(r"C:\Users\pro pc\Desktop\python1\AI\excel\excel acctualisados\TheBostonHousingDataset.xlsx")
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

#ESCALADOR  OI NORMILIZADOR
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

#Construccion red neuronal

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1],activation ='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', 
              loss ='mean_squared_error')
r2_callback = R2Callback(train_data=(X_train, y_train),val_data=(X_test,y_test))

#entrenamiento del modelo
early_stopin_monitor = EarlyStopping (monitor = 'val_loss',patience  = 20)
historial = model.fit(X_train,y_train,validation_data = (X_test, y_test),epochs=1000,batch_size=30,callbacks=[early_stopin_monitor,r2_callback]) 
perdida = model.evaluate (X_test,y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print(f'el valor de r2 es: {r2}')
print(f'el error es :{error}')

#Graficacion de perdidas durante el entrenamiento
plt.figure(figsize=(12,6))
plt.plot(historial.history['loss'],label='perdida entrenamiento')
plt.plot(historial.history['val_loss'],label='perdiada de validacion')
plt.title("TEST LOSS")
plt.xlabel("epoch") 
plt.ylabel("loss")
plt.legend()
plt.show()



#Grafica de coefisiente de determinacion de r2 

plt.figure(figsize=(12,6))
plt.plot(r2_callback.val_r2,label='validacion de r2')
plt.plot(r2_callback.train_r2,label='enrtenamiendo de r2')
plt.title("Grafica de coefisiente de determinacion de r2")
plt.xlabel("epoch") 
plt.ylabel("r2")
plt.legend()
plt.show()



#Graficar valores predichos vs Valores Reales

plt.figure(figsize=(12,6))
plt.scatter(y_test,y_predict)
plt.plot([y_test.min(),y_test.max()],[y_predict.min(),y_predict.max()])
plt.title("Graficar valores predichos vs Valores Reales")
plt.xlabel("real") 
plt.ylabel("predicho")
plt.tight_layout()
plt.show()


#Graficar valores Residuales
residuales = y_test-y_predict.flatten()
plt.figure(figsize=(12,2))
plt.scatter(y_predict,residuales)
plt.hlines(y=0,xmin=y_predict.min(),xmax=y_predict.max())
plt.title("Grafica valores residuales")
plt.ylabel("Residual")
plt.xlabel("predicioness")
plt.show()

#histograma residuales.
plt.figure(figsize=(12,2))
plt.hist(residuales,bins=30,edgecolor="orange")
plt.title("Histograma residuales.")
plt.xlabel('Residuales')
plt.ylabel("Frequencia")
plt.show()



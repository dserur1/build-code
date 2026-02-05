import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

#This is the data frame that is being used to get the information from
df = pd.read_excel(
    r"C:\Users\pro pc\Desktop\python1\AI\excel\excel acctualisados\TheBostonHousingDataset.xlsx")
#test estep
print(df.head())

#listas

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

#Visualisasion de datos

plt.figure(figsize=(15,9))
# hacer 2 graficas 1 precio vs #de habitaciones y 2 precio vs impuestos
plt.subplot(1,2,1)
plt.scatter(RM , MEDV , color='green')
plt.title("PRECIO VS HABITACIONES")
plt.xlabel("numero de habitraciones")
plt.ylabel("PRECIO")
#grafica precio vs impuestos
plt.subplot(1,2,2)
plt.scatter(TAX , MEDV , color='blue')
plt.title("precio vs impuesto")
plt.xlabel("impestos")
plt.ylabel("precio")
plt.tight_layout()
plt.show()

# ASIGNACION DE VARIABLES INDEPENDIENTES Y DEPENDIENTES.
y = np.array(MEDV)
x = np.array(list(zip(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)))

# MODELO DE INTELIGENCIA ARTIFICIAL
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ENTRENAMIENTO DE MODELO
Modelo = DecisionTreeRegressor(max_depth=5,random_state=42)
Modelo.fit(X_train,y_train)   

# REALIZANDO PREDICCIONES
y_predict = Modelo.predict(X_test)
print(f"este es tu prediccion de presio: {round(y_predict[0],2)}")

# EVALUACION DEL MODELO
r2 = r2_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict) 
print(f'este es el herror.. {round(error,2)}, esta es tu preciocion.. {round(r2,2)}') 

# VISUALISASION DEL ARBOL DE DESISOINES
plt.figure(figsize=(20,16))
vector = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
tree.plot_tree(Modelo,feature_names = vector,filled = True, rounded = True )
plt.title('ARBOL DE DESISIONES')
plt.savefig('tree.png', dpi=1080 , bbox_inches = 'tight' )
plt.show() 

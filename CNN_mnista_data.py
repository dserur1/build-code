# Se necesita creacion de codigo con modelo cnn para reconocer patrones en las imagenes y comparar con texto, evaluar modelo,
# grafica de predicciones corectas e incorectas,definir arquitectura de la cnn , visualir istorial de entrenamiento ,crear matirz de confucion
# crear curvas ROC y AUC ,gardar modelo , creacion de funcion para cargar imagenes y etiquetas desde la carpeta.


#librerias
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,roc_curve,auc


#funciones


def load_data (data_path,data_type):
    label_dir = os.path.join(data_path, f'{data_type}_labels')
    image_dir = os.path.join(data_path, f'{data_type}_images')
    labels = []
    images = []
    for file_name in sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0])):
        #cargar imagen
        image_path = os.path.join(image_dir,file_name)
        image = Image.open(image_path).convert('L')
        image = np.array(image) / 255.0  # Normalizar la imagen
        images.append(image)
         #Cagar etiqueta 
        label_path = os.path.join(label_dir,os.path.splitext (file_name) [0] + '.txt' )
        with open(label_path ,'r') as f :
            label = int(f.read().strip()) 
        labels.append(label)

    labels = np.array(labels) 
    images = np.array(images).reshape(-1,28,28,1)
    return (labels,images)

#Funcion para graficar predicciones corectas & incorectas

def graficador_de_evaluacion(label,images,predicciones,maximo):
    indcie_correcto= np.where(predicciones==label)[0]
    indice_incorrecto= np.where(predicciones!=label)[0]
    plt.figure(figsize=(16,8))
    for i,correctas in enumerate (indcie_correcto[:maximo]):
     plt.subplot (1,maximo,i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(images[correctas].reshape(28,28),cmap = plt.cm.binary)
     plt.xlabel(f'pred: {predicciones[correctas]}(true:{label [correctas]})')

    for i,incorrecto in enumerate (indice_incorrecto[:maximo]):
        plt.subplot(2,maximo,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[incorrecto].reshape(28,28),cmap = plt.cm.binary) 
        plt.xlabel(f'pred: {predicciones[incorrecto]}(true:{label [incorrecto]})')
    plt.show()                              
 

# codigo base 
# los datos que nesesitamos
path_mnist = r'C:\Users\pro pc\Desktop\python1\AI\DEEP_LERNING_IA\mnist_data'

#variables de entrenamiento y testeo
train_labels, train_images  =  load_data(path_mnist,'train')
test_labels, test_images = load_data(path_mnist,'test') 
print(f"train_labels:{train_labels.shape}")
print(f"train_images:{train_images.shape}")
print(f"test_labels:{test_labels.shape}")
print(f"test_images:{test_images.shape}")


# Comprobacion de imagen

imagen = 25
plt.figure(figsize=(10,10))
for i in range (imagen):
    plt.subplot (5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(28,28),cmap = plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# Definir la arquitectura de la CNN
print('\nCreando arquitectura de la CNN ... ')
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss ='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
print('\nEntrenando el modelo ... ')
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


# Guardar el modelo entrenado
print("\nModelo guardado como 'modelo_mnist.keras'.")
model.save('modelo_mnist.keras')


# Evaluar el modelo
print('\nEvaluando el modelo ... ')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Precisión en el conjunto de prueba: {test_acc}")

# visualisasion de istorial de entrenamiento

plt.figure(figsize=(12,12))
plt.subplot (2,1,1)
plt.plot(history.history['val_accuracy'],label='presision de validacion')
plt.plot(history.history['accuracy'],label='precicion de entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')

#continaur con grafica de perdidas

plt.subplot(2,1,2)
plt.plot(history.history['val_loss'],label='perdida de validacion')
plt.plot(history.history['loss'],label='perdida de entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Perdidas')
plt.legend(loc='upper right')
plt.show()

#modelo de predicciones

prediccion = model.predict(test_images)
prediccion_label = np.argmax(prediccion,axis=1)

#Matriz de confusion
mc = confusion_matrix(test_labels,prediccion_label)
display = ConfusionMatrixDisplay(confusion_matrix=mc,display_labels = range(10))
display.plot(cmap=plt.cm.Reds)
plt.show()

#Curvas ROC y AUC
fp = dict()
vp = dict()
a_curva = dict()
for i in range(10):
    fp[i],vp[i], _ = roc_curve(test_labels == i, prediccion[:,i])
    a_curva[i] = auc(fp[i],vp[i])
plt.figure()
colors = plt.cm.get_cmap('tab10', 10)
for i, color in zip(range(10), colors.colors):
    plt.plot(fp[i], vp[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {a_curva[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

#Grafica representa predicciones correctas y incorrectas
graficador_de_evaluacion(test_labels,test_images, prediccion_label,10)

#libreria 
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog as fi
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#fuciones
def upload_image():
    root = tk.Tk()
    root.withdraw()
    path = fi.askopenfilename()
    return(path)


def procesamiento_imagen(path):
    imagen1 = Image.open(path).convert("L")
    imagen1 = imagen1.resize((28,28))
    # Normalizar la imagen
    imagen1 = np.array(imagen1) / 255.0  
    imagen1 = imagen1.reshape(-1,28,28,1)
    return (imagen1)


     

#codigo base



#modelo entrenado
model=load_model('modelo_mnist.keras')
print('modelocargado')


#importacion de imagnes

key = True
while key:
    path = upload_image()
    if path:
        imagen = procesamiento_imagen(path)
        print(imagen)
        key = False
        #modelo prediccion
        prediccion = model.predict(imagen)
        prediccion_label = np.argmax(prediccion,axis=1)
        print(prediccion)
        print(prediccion_label)
        #plot de graficos
        plt.imshow(imagen.reshape(28,28),cmap = plt.cm.gray)
        plt.title(f'pred: {prediccion_label})')
        plt.show()
    else:
        print("pleas select a paht now asshole!")
        

         




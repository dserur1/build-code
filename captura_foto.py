
#diseno de codigo para para enceder camara tomar una foto
# prender la cmara tomar la foto y dowload la foto a aun archibo
import cv2
def tomada_de_foto(foto1 = "foto.jpg"):
    #camra prendida
    camon = cv2.VideoCapture(0)
    if not camon.isOpened():
        print("the cam is off fuker")
        return
    print("press's space bar to capture image, and esc to exit")
    while True:
        # capura de frame
        ret,frame = camon.read()
        if not ret:
            print(" your camra is fuked up")
            break
        cv2.imshow("image captured",frame)
        takepic = cv2.waitKey(1)
        if takepic == 13:
            cv2.imwrite(foto1,frame)
            print("the image is saved in you ass and named",foto1 )
            break
        elif takepic == 48:
            print("your pic is deleted")
    camon.release()
    cv2.destroyAllWindows()
tomada_de_foto()         



    

 





    
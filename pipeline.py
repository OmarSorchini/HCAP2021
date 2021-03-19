'''
Programa que cuenta con la función pipeline que reproduce los pasos para procesar una imágen mediante convoluciones y maxpooling
Autores:
    * David Rodríguez Fragoso A01748760
    * Israel Sánchez Miranda A01378705
    * José Miguel García Gurtubay Moreno A01373750
    * Miguel Angel Juárez Dorantes A01753328
    * Omar Rodrigo Sorchini Puente A01749389
    * Paola Dorantes Calderón A01653108
19/03/2021
'''
#Bibliotecas importadas
import cv2                 #OpenCV
import numpy as np         #Numpy
import maxpooling as mp    #Programa de maxpooling
import convolucion as conv #Programa de convolución

#Función de pipeline
def pipeline(img):
    '''
    Función que se encarga de reproducir el pipline para procesar una imágen dada

    Parámetros:
    * img = imagen dada a la función para su posterior procesamiento

    Retornos:
    * No retorna nada, sin embargo, crea una imagen procesada por el pipeline y la guarda en el directorio donde se encuentre el programa
    '''

    #Variables
    IRGB = cv2.imread(img)                          #Imagen leída a color
    IMGS = cv2.cvtColor(IRGB, cv2.COLOR_BGR2GRAY)   #Imagen leída en escala de grises
    Kernel1 = 1/256*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,2,1]])
    Kernel2 = np.array([[2,2,4,2,2],[1,1,2,1,1],[0,0,0,0,0],[-1,-1,-2,-1,-1],[-2,-2,-4,-2,-2]])

    IR = conv.convolucion(IMGS, Kernel1) #Primera iteración de las convoluciones a la imagen resultante
    IR = mp.maxpooling(IR)         #Primera iteración del maxpool a la imagen resultante
    #Segunda iteración de convoluciones con un Kernel diferente y del maxpooling
    IR = conv.convolucion(IR, Kernel2)
    IR = mp.maxpooling(IR)

    cv2.imwrite('ICMPCS_Omar.jpg', IR) #Se guarda la imágen nueva

#Probando la función
pipeline('014.jpg') #Se le manda una imágen escogida a la función pipeline

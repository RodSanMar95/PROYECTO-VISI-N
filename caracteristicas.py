# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:33:33 2024

@author: rodri
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Lista de imágenes para procesar
imagenes = np.array(['imagen3.jpg', 'imagen4.jpg'])

# Bucle para procesar cada imagen
for pic in imagenes:
    img = cv2.imread(pic)
    if img is None:
        print(f'No se encuentra la imagen: {pic}\n')
        sys.exit(1)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Umbralización con Otsu
   
    th, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (43,43))

    img_cierre = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ee)
   
    plt.subplot(2,2,1), plt.title ('Gray'), 
    plt.imshow(gray,'gray', vmin=0, vmax=225), plt.axis(False)
    plt.subplot(2,2,2), plt.title ('Umbral:{}'.format(th)), 
    plt.imshow(img_cierre,'gray', vmin=0, vmax=225), plt.axis(False)
    
    plt.show()
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 50:  # Ignorar contornos pequeños
            continue
        
        # Calcular características
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:  # Evitar división por cero
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        theta = 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))
        circularity = 4 * np.pi * area / (perimeter**2)
        compacity = area / (perimeter**2)
        
        rect = cv2.minAreaRect(contour)
        alto_ancho = rect[1]
        rectangularity = area/(alto_ancho[0]*alto_ancho[1])

        # Dibujar contorno y centroide en la imagen
        cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
        
        #Marcamos la punta a partir de su circularidad, compactidad, rectangularidad 
        if(area >= 10000 and 0.33 <= circularity <= 0.48 
           and 0.025 <= compacity <= 0.04 and rectangularity < 0.58):
            # Obtener y dibujar el rectángulo delimitador alrededor del contorno.
           xb, yb, wb, hb = cv2.boundingRect(contour)
           cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
           cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
           cv2.rectangle(img, (xb, yb), (xb+wb, yb+hb), (0, 0, 255), 2)
           
        
        # Imprimir los datos calculados en la consola.
        print(f"Objeto nº{i+1}")
        print(f"{rect}")
        print(f"Perímetro-> {perimeter:.3f}")
        print(f"Área-> {area:.3f}")
        print(f"Coordenadas del CdG-> ({cX}, {cY})")
        print(f"Ángulo de los ejes principales de inercia-> {theta:.3f}")
        print(f"Grado de circularidad-> {circularity:.3f}")
        print(f"Compacidad-> {compacity:.3f}")
        print(f"Rectangularidad-> {rectangularity:.3f}\n\n")

        
        # Etiquetar el objeto en la imagen
        cv2.putText(img, f'{i+1}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Configurar la ventana
    window_name = 'Imagen'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    
    
    cv2.destroyAllWindows()


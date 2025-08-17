# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:43:26 2024

@author: rodri
"""

import cv2
import numpy as np

import sys

# Nombre del archivo de video
video_file = 'video2.mp4'

# Abrir el video
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f'Error al abrir el archivo de video: {video_file}')
    sys.exit(1)

# Determinar las propiedades del video de salida
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Configuración del codificador y creación del objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, frame_rate, (frame_width, frame_height))

# Configurar la ventana
window_name = 'Frame'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

# Bucle para procesar cada fotograma del video
while True:
    ret, frame = cap.read()
    if not ret:
        print('No se pudo leer el fotograma del video o fin del video')
        break

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    th, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Cierre morfológico para eliminar pequeños agujeros en los objetos
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (43,43))
    img_cierre = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ee)

    # Encontrar contornos
    contours, _ = cv2.findContours(img_cierre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Procesamiento de contornos y otras operaciones como en el código original
    # ...
    # Asegúrate de reemplazar `img` con `frame` en las operaciones de dibujo
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
        
        """
        # Dibujar contorno y centroide en la imagen
        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)"""
        
        #Marcamos la punta a partir de su circularidad, compactidad, rectangularidad 
        if(1000 <= area <= 10000 and 0.33 <= circularity <= 0.48 
           and 0.025 <= compacity <= 0.04 and rectangularity < 0.58):
            # Obtener y dibujar el rectángulo delimitador alrededor del contorno.
           xb, yb, wb, hb = cv2.boundingRect(contour)
           cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
           cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
           cv2.rectangle(frame, (xb, yb), (xb+wb, yb+hb), (0, 0, 255), 2)
           
        
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
        
        # Detectar las rectas mediante la transformada de Hough
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=10, maxLineGap=0)
        
        # Dibujar las líneas detectadas en la imagen
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                size = round(np.sqrt((x2-x1)**2 + (y2-y1)**2), 2)
                angle = round(np.arctan2((y2-y1), (x2-x1)) * 180/np.pi, 2)
                
                # Filtramos rectas horizontales y verticales
                if 20 <= size <= 1000:  
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    print(f'Línea {i+1}: Tamaño={size}, Ángulo={angle}')

        
        # Etiquetar el objeto en la imagen
        cv2.putText(frame, f'{i+1}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Suponiendo que "frame" es el fotograma modificado que quieres guardar
    out.write(frame)  # Guardar el fotograma en el archivo de salida
    
    # Configurar la ventana
    window_name = 'Imagen'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    # Mostrar el resultado en una ventana
    cv2.imshow('Frame', frame)

    # Pausa para que la ventana se actualice y compruebe si el usuario presionó la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto cap y cerrar todas las ventanas abiertas
cap.release()
out.release()  # No olvides liberar el objeto VideoWriter
cv2.destroyAllWindows()

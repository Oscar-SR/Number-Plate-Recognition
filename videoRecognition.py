import cv2
import numpy as np
import os
import pytesseract
import platform

sistema = platform.system()

if sistema == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
elif sistema == "Linux":
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
elif sistema == "Darwin":  # macOS
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
else:
    raise EnvironmentError("Sistema operativo no soportado para configurar Tesseract automáticamente")

# Declaración de constantes. Pueden ser ajustables según el caso
ASPECT_RATIO_MIN = 3
ASPECT_RATIO_MAX = 5.5
AREA_MIN = 1000
AREA_MAX = 30000
TEXTO_MIN = 5
TEXTO_MAX = 9

# Pedir al usuario la ruta o nombre del archivo de video
video = input("Introduce la ruta del vídeo: ")

# Abrir el video
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Configurar la ventana para pantalla completa
cv2.namedWindow("Matrículas Detectadas", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Matrículas Detectadas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Obtener FPS, ancho y alto del vídeo original
fps = cap.get(cv2.CAP_PROP_FPS)
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Guardar el video en formato MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificador para formato .mp4
nombre_video, extension = os.path.splitext(video)
video_procesado = f"{nombre_video} (processed){extension}"
video_writer = cv2.VideoWriter(video_procesado, fourcc, fps, (ancho, alto))

while True:
    # Leer un fotograma del video
    ret, frame = cap.read()
    
    if not ret:
        print("No se le leyeron más fotogramas del video.")
        break

    # Convertimos a escala de grises y aplicamos filtro Gaussiano
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (9, 9), 1)

    # Aplicar filtro Sobel horizontal
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

    # Aplicar umbralización binaria con el método Otsu
    _, threshold = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Eliminamos ruido de la imagen con dilatación y erosión
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    threshold = cv2.erode(threshold, kernel, iterations=1)

    # Detección de bordes con Canny
    sigma = 0.33
    mediana = np.median(blur)
    lower = int(max(0, (1.0 - sigma) * mediana))
    upper = int(min(255, (1.0 + sigma) * mediana))
    bordes = cv2.Canny(threshold, lower, upper)

    # Detección de contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        # Calcular el rectángulo
        x, y, w, h = cv2.boundingRect(c)

        # Calcular aspect_ratio del rectángulo
        aspect_ratio = w / float(h)
        
        # Verificar las condiciones de relación de aspecto
        if ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX:
            # Calcular área del rectándulo
            area = w * h
            
            # Verificar las condiciones de área
            if AREA_MIN <= area <= AREA_MAX:
                 # Obtenemos la matrícula en gris para binarizarla
                matricula = gris[y:y+h, x:x+w]

                # Binarizamos la matrícula para facilitar el reconocimiento del texto
                _, matricula_bin = cv2.threshold(matricula, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Obtenemos el texto
                texto = pytesseract.image_to_string(matricula_bin, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') # Restringimos los carácteres válidos
                texto = texto.strip().replace("\n", "") # Elimina los espacios en blanco por delante y por detrás, y quita los saltos de línea

                if texto and TEXTO_MIN <= len(texto) <= TEXTO_MAX:
                    # Escribir texto sobre el contorno y dibujar contorno
                    cv2.putText(frame, f"Matricula: {texto}", (x, y - 5), 1, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Escribir el fotograma procesado en el archivo de salida
    video_writer.write(frame)

    # Mostrar el video con las matrículas detectadas
    cv2.imshow("Matrículas Detectadas", frame)

    # Controlar la velocidad del video con un pequeño retraso
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

# Liberar recursos
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Video procesado guardado como '{video_procesado}'.")

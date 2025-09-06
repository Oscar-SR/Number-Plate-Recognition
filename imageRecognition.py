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

# 1. CARGAR LA IMAGEN

ruta = input("Introduce la ruta de la imagen: ")
imagen = cv2.imread(ruta) # Ruta de la imagen

if imagen is None:
    print("Error: No se pudo cargar la imagen. Deteniendo ejecución.")
    exit() # Detiene la ejecución del programa

print("Imagen cargada correctamente.")
cv2.imshow("Imagen original", imagen)
cv2.waitKey(0)  # Espera a que el usuario presione una tecla antes de continuar
cv2.destroyAllWindows()  # Cierra todas las ventanas al finalizar

# 2. PREPROCESAMIENTO DE LA IMAGEN

# Convertimos a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar filtro Gaussiano
blur = cv2.GaussianBlur(gris, (9, 9), 1)

# Aplicar filtro Sobel horizontal para detectar los bordes horizontales
sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

# Aplicar un filtro binario
_, threshold = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Eliminamos ruido de la imagen y facilitamos la detección de contornos
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
threshold = cv2.dilate(threshold, kernel, iterations=1)
threshold = cv2.erode(threshold, kernel, iterations=1)

cv2.imshow("Imagen preprocesada", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. FILTRO CANNY

# Calculamos el umbral inferior y superior
sigma = 0.33
mediana = np.median(blur)
lower = int(max(0, (1.0 - sigma) * mediana))
upper = int(min(255, (1.0 + sigma) * mediana))

# Detección de bordes con Canny
bordes = cv2.Canny(threshold, lower, upper)
bordes = cv2.dilate(bordes, None, iterations=1)
bordes = cv2.erode(bordes, None, iterations=1)

cv2.imshow("Filtro Canny", bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. DETECCIÓN DE CONTORNOS

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
        # Mostrar matrícula
        cv2.imshow("Matrícula detectada", matricula_bin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Escribir texto sobre el contorno y dibujar contorno
        cv2.putText(imagen, f"Matricula: {texto}", (x, y - 5), 1, 1, (0, 255, 0), 2)
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Matriculas detectadas", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen procesada
nombre_imagen, extension = os.path.splitext(ruta)
imagen_procesada = f"{nombre_imagen} (processed){extension}"
cv2.imwrite(imagen_procesada, imagen)
print(f"Imagen procesada guardada en: {imagen_procesada}")
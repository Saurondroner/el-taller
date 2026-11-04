import cv2
import time
import os
import winsound

# Ajustes
CAMARA = 0                 # Prueba 1 o 2 si tu cámara no aparece en el índice 0
AREA_MINIMA = 1500         # Sube este valor para ignorar movimientos pequeños
INTERVALO_AVISOS = 2       # Segundos entre pitidos

# Abrir la cámara
camara = cv2.VideoCapture(CAMARA, cv2.CAP_DSHOW)

if not camara.isOpened():
    print("No se ha podido abrir la cámara.")
    print("Cierra la aplicación Cámara de Windows y prueba CAMARA = 1.")
    raise SystemExit

# Detector de fondo: permite adaptarse poco a poco a la escena
detector_fondo = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=40,
    detectShadows=True
)

ultimo_aviso = 0

print("---------------------------------------")
print(" DETECTOR DE MOVIMIENTO EN FUNCIONAMIENTO")
print(" Pulsa Q en la ventana de vídeo para salir")
print("---------------------------------------")

while True:
    correcto, imagen = camara.read()

    if not correcto:
        print("No se puede leer la imagen de la cámara.")
        break

    mascara = detector_fondo.apply(imagen)
    mascara = cv2.GaussianBlur(mascara, (7, 7), 0)
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)

    mascara = cv2.dilate(mascara, None, iterations=2)
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    movimiento = False

    for contorno in contornos:
        if cv2.contourArea(contorno) < AREA_MINIMA:
            continue

        movimiento = True
        x, y, ancho, alto = cv2.boundingRect(contorno)
        cv2.rectangle(
            imagen,
            (x, y),
            (x + ancho, y + alto),
            (0, 0, 255),
            2
        )

    if movimiento:
        cv2.putText(
            imagen,
            "MOVIMIENTO DETECTADO",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if time.time() - ultimo_aviso >= INTERVALO_AVISOS:
            winsound.Beep(1000, 200)
            ultimo_aviso = time.time()
    else:
        cv2.putText(
            imagen,
            "ZONA VIGILADA",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("MI CAMARA DE SEGURIDAD INTELIGENTE", imagen)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()
print("Detector cerrado correctamente.")

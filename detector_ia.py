
import cv2
from ultralytics import YOLO

print("====================================")
print("     EPIK4L - DETECTOR DE IA")
print("====================================")
print("Cargando inteligencia artificial...")

# Cargar modelo YOLO11 pequeño
modelo = YOLO("yolo11n.pt")

# Buscar una cámara disponible
camara = None

for numero in range(5):
    prueba = cv2.VideoCapture(numero, cv2.CAP_DSHOW)

    if prueba.isOpened():
        correcto, imagen = prueba.read()

        if correcto and imagen is not None:
            camara = prueba
            print(f"Camara encontrada: {numero}")
            break

    prueba.release()

if camara is None:
    print("ERROR: No se encuentra ninguna camara.")
    print("Comprueba que la camara esta conectada.")
    input("Pulsa ENTER para salir...")
    exit()

print("INTELIGENCIA ARTIFICIAL ACTIVADA")
print("Pulsa Q para cerrar el programa.")

while True:
    correcto, imagen = camara.read()

    if not correcto:
        print("No se recibe imagen de la camara.")
        break

    # Reconocer objetos
    resultados = modelo.predict(
        source=imagen,
        imgsz=640,
        conf=0.35,
        verbose=False
    )

    # Dibujar recuadros y nombres
    imagen_ia = resultados[0].plot()

    # Mostrar imagen
    cv2.imshow("EPIK4L - Inteligencia Artificial", imagen_ia)

    # Salir pulsando Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()

print("Programa finalizado.")

"""EcoSort — clasificación de material EN VIVO con la cámara (OpenCV).

Abre la webcam, corre los 3 detectores sobre cada frame y superpone el material
detectado + las probabilidades. Presiona 'q' para salir.

También acepta una imagen o un video como fuente (para probar sin webcam), y una
opción --save para guardar el frame anotado (útil en entornos sin pantalla).

Ejemplos:
    python camara.py                          # webcam (framework TF)
    python camara.py --framework torch        # webcam con PyTorch
    python camara.py --source foto.jpg --save salida.jpg
    python camara.py --source video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import ecosort_infer as infer

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def annotate(frame, probs, material, confidence):
    """Dibuja el resultado sobre el frame (BGR de OpenCV)."""
    height, width = frame.shape[:2]
    etiqueta = f'{material.upper()} {confidence:.0%}' if material else 'INDEFINIDO'
    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.putText(frame, f'EcoSort: {etiqueta}', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    y = 65
    for m in infer.MATERIALS:
        cv2.putText(frame, f'{m}: {probs[m]:.0%}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
    return frame


def predict_bgr(models, framework, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    probs = infer.predict_probs(models, framework, rgb)
    material, confidence = infer.classify(probs)
    return probs, material, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description='EcoSort — clasificación de material con la cámara (OpenCV).')
    parser.add_argument('--framework', choices=['tf', 'torch'], default='tf')
    parser.add_argument('--source', default='0', help='0 = webcam; o ruta a imagen/video.')
    parser.add_argument('--save', default=None, help='Guardar el frame anotado en esta ruta (sin pantalla).')
    args = parser.parse_args()

    print('Cargando modelos...')
    models = infer.load_models(args.framework)

    # --- Fuente = imagen estática ---
    if args.source != '0' and Path(args.source).suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f'No se pudo leer la imagen {args.source}')
        probs, material, confidence = predict_bgr(models, args.framework, frame)
        annotate(frame, probs, material, confidence)
        if args.save:
            cv2.imwrite(args.save, frame)
            print(f'Resultado: {material} ({confidence:.0%}) -> guardado en {args.save}')
        else:
            cv2.imshow('EcoSort', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # --- Fuente = webcam (int) o video (ruta) ---
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f'No se pudo abrir la fuente {args.source} (¿hay cámara disponible?).')

    print("Cámara abierta. Presiona 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        probs, material, confidence = predict_bgr(models, args.framework, frame)
        annotate(frame, probs, material, confidence)
        cv2.imshow('EcoSort', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

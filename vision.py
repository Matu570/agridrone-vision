import cv2
import numpy as np


class InvalidImageError(Exception):
    """Se lanza cuando OpenCV no puede leer o guardar una imagen."""


def detect_and_count_crops(source_path: str, output_path: str) -> tuple[str, int]:
    """Detecta cultivos verdes con OpenCV y retorna (ruta procesada, conteo)."""
    image = cv2.imread(source_path)
    if image is None:
        raise InvalidImageError(
            "No se pudo leer la imagen. El archivo es inválido o está dañado."
        )

    try:
        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    except cv2.error as exc:
        raise InvalidImageError("La imagen no tiene un formato de color válido.") from exc

    # Tonos verdes y amarillentos típicos de vegetación.
    lower_yellow_green = np.array([20, 35, 35])
    upper_yellow_green = np.array([40, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(yellow_mask, green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Opening: separa plantas que se tocan (erode + dilate).
    eroded = cv2.erode(mask, kernel, iterations=2)
    opened = cv2.dilate(eroded, kernel, iterations=2)

    # Closing: rellena huecos en zonas de sombra (dilate + erode).
    dilated = cv2.dilate(opened, kernel, iterations=2)
    closed = cv2.erode(dilated, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    image_area = float(height * width)
    min_area = max(150.0, image_area * 0.00045)
    max_area = image_area * 0.035

    annotated = image.copy()
    total_detections = 0
    box_color = (0, 255, 0)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)
        total_detections += 1
        cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), box_color, 2)
        cv2.putText(
            annotated,
            str(total_detections),
            (x, max(y - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA,
        )

    saved = cv2.imwrite(output_path, annotated)
    if not saved:
        raise InvalidImageError("No se pudo guardar la imagen procesada.")

    return output_path, total_detections

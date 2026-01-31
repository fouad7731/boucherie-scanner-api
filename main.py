from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)  # Permet les appels depuis ton app Vercel


def order_points(pts):
    """Ordonne les 4 points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left = plus petite somme, bottom-right = plus grande somme
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right = plus petite difference, bottom-left = plus grande difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """Applique une transformation de perspective pour avoir une vue de dessus"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculer les dimensions du nouveau document
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Points de destination (rectangle parfait)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Calculer la matrice de transformation et appliquer
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def detect_document(image):
    """Detecte les bords du document dans l'image"""
    orig = image.copy()

    # Redimensionner pour le traitement (plus rapide)
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flou gaussien pour reduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detection des bords avec Canny - seuils adaptatifs
    median_val = np.median(blurred)
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))
    edged = cv2.Canny(blurred, lower, upper)

    # Dilatation pour fermer les contours
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(edged, kernel, iterations=1)

    # Trouver les contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None

    for contour in contours:
        # Approximer le contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Si le contour a 4 points et une aire significative
        if len(approx) == 4:
            # Verifier que l'aire est suffisante (au moins 10% de l'image)
            area = cv2.contourArea(approx)
            img_area = image.shape[0] * image.shape[1]
            if area > img_area * 0.1:
                doc_contour = approx
                break

    if doc_contour is not None:
        # Remettre a l'echelle originale
        return doc_contour.reshape(4, 2) * ratio

    return None


def trim_white_borders(image, threshold=250, margin=5):
    """Supprime les bordures blanches autour du document"""
    # Convertir en niveaux de gris si necessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Inverser (le contenu devient blanc, le fond devient noir)
    inverted = cv2.bitwise_not(gray)

    # Seuil pour detecter le contenu (tout ce qui n'est pas blanc)
    _, thresh = cv2.threshold(inverted, 255 - threshold, 255, cv2.THRESH_BINARY)

    # Trouver les pixels non-zero (le contenu)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        # Obtenir le rectangle englobant
        x, y, w, h = cv2.boundingRect(coords)

        # Ajouter une petite marge
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        # Cropper l'image
        if len(image.shape) == 3:
            return image[y:y+h, x:x+w]
        else:
            return image[y:y+h, x:x+w]

    return image


def enhance_document(image):
    """Ameliore l'image pour une meilleure lisibilite"""
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Amelioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Debruitage leger
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 8, 7, 21)

    # Sharpening leger
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Reconvertir en couleur (BGR) pour compatibilite
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # Supprimer les bordures blanches
    result = trim_white_borders(result)

    return result


def process_document(image_base64, enhance=True):
    """Pipeline complet de traitement du document"""
    # Decoder l'image base64
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    image_bytes = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Impossible de decoder l'image")

    # 1. Detecter le document
    doc_corners = detect_document(image)

    detected = doc_corners is not None

    if doc_corners is not None:
        # 2. Appliquer la transformation de perspective
        warped = four_point_transform(image, doc_corners)
    else:
        # Pas de document detecte - utiliser l'image telle quelle
        warped = image

    # 3. Ameliorer l'image (optionnel)
    if enhance:
        result = enhance_document(warped)
    else:
        result = warped

    # 4. Encoder en base64 (JPEG haute qualite)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    _, buffer = cv2.imencode('.jpg', result, encode_params)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'processedImage': f"data:image/jpeg;base64,{result_base64}",
        'documentDetected': detected,
        'originalWidth': image.shape[1],
        'originalHeight': image.shape[0],
        'processedWidth': result.shape[1],
        'processedHeight': result.shape[0],
    }


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de sante pour Railway"""
    return jsonify({'status': 'ok', 'service': 'boucherie-scanner-api'})


@app.route('/process', methods=['POST'])
def process():
    """Endpoint principal pour traiter une image"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Image manquante'}), 400

        enhance = data.get('enhance', True)
        result = process_document(data['image'], enhance=enhance)

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        print(f"Erreur traitement: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint pour juste detecter les coins (sans transformation)"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Image manquante'}), 400

        # Decoder l'image
        image_base64 = data['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Image invalide'}), 400

        # Detecter
        corners = detect_document(image)

        if corners is not None:
            return jsonify({
                'success': True,
                'detected': True,
                'corners': corners.tolist()
            })
        else:
            return jsonify({
                'success': True,
                'detected': False,
                'corners': None
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

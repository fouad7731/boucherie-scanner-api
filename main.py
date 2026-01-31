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
    """Detecte les bords du document dans l'image - detection papier blanc"""
    orig_h, orig_w = image.shape[:2]

    # Redimensionner pour le traitement (plus rapide)
    ratio = orig_h / 500.0
    resized = cv2.resize(image, (int(orig_w / ratio), 500))

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    doc_contour = None

    # === METHODE 1: Detection du papier blanc ===
    # Le papier est generalement plus clair que le fond
    # Seuil haut pour isoler les zones blanches/claires
    _, white_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    # Nettoyer le masque
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        for epsilon in [0.02, 0.03, 0.04, 0.05, 0.06]:
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)

            if len(approx) == 4:
                area = cv2.contourArea(approx)
                img_area = resized.shape[0] * resized.shape[1]

                # Au moins 10% de l'image
                if area > img_area * 0.10:
                    doc_contour = approx
                    break

        if doc_contour is not None:
            break

    # === METHODE 2: Canny si methode 1 echoue ===
    if doc_contour is None:
        for canny_low, canny_high in [(30, 100), (50, 150), (75, 200)]:
            edged = cv2.Canny(blurred, canny_low, canny_high)

            kernel = np.ones((3, 3), np.uint8)
            edged = cv2.dilate(edged, kernel, iterations=2)

            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    img_area = resized.shape[0] * resized.shape[1]

                    if area > img_area * 0.10:
                        doc_contour = approx
                        break

            if doc_contour is not None:
                break

    # === METHODE 3: Convex Hull du plus grand contour ===
    if doc_contour is None:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) == 4:
                area = cv2.contourArea(approx)
                img_area = resized.shape[0] * resized.shape[1]
                if area > img_area * 0.10:
                    doc_contour = approx

    if doc_contour is not None:
        # Remettre a l'echelle originale
        return doc_contour.reshape(4, 2) * ratio

    return None


def trim_to_content(image, margin=5):
    """Crop au contenu en detectant le TEXTE NOIR (approche inverse)"""
    print(f"[TRIM] Input: {image.shape[1]}x{image.shape[0]}")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    print(f"[TRIM] Luminosite - min:{gray.min()} max:{gray.max()} mean:{gray.mean():.0f}")

    # === APPROCHE: Detecter le TEXTE (pixels sombres) ===
    # Le texte est noir/gris fonce (< 150)
    # Les marges grises sont ~196, le papier blanc ~230
    # Donc tout ce qui est < 150 = texte

    _, text_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text_pixels = np.sum(text_mask > 0)
    print(f"[TRIM] Pixels texte (<150): {text_pixels} = {100*text_pixels/(h*w):.1f}%")

    # Dilater le texte pour connecter les caracteres et lignes proches
    kernel = np.ones((10, 10), np.uint8)
    text_dilated = cv2.dilate(text_mask, kernel, iterations=3)

    # Trouver le bounding box du texte
    coords = cv2.findNonZero(text_dilated)

    if coords is not None and len(coords) > 100:
        x, y, bw, bh = cv2.boundingRect(coords)
        print(f"[TRIM] Bounding box texte: ({x},{y}) {bw}x{bh}")

        # Ajouter une marge
        x = max(0, x - margin)
        y = max(0, y - margin)
        bw = min(w - x, bw + 2 * margin)
        bh = min(h - y, bh + 2 * margin)

        result = image[y:y+bh, x:x+bw] if len(image.shape) == 3 else gray[y:y+bh, x:x+bw]
        print(f"[TRIM] Output: {result.shape[1]}x{result.shape[0]}")
        print(f"[TRIM] Reduction: {w}x{h} -> {result.shape[1]}x{result.shape[0]}")
        return result

    print(f"[TRIM] Pas de texte detecte - retour image originale")
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

    # Cropper au plus pres du contenu
    result = trim_to_content(result)

    return result


def process_document(image_base64, enhance=True):
    """Pipeline complet de traitement du document"""
    debug_info = {}

    # Decoder l'image base64
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    image_bytes = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Impossible de decoder l'image")

    debug_info['original_size'] = f"{image.shape[1]}x{image.shape[0]}"
    print(f"[DEBUG] Image originale: {image.shape[1]}x{image.shape[0]}")

    # 1. Detecter le document
    doc_corners = detect_document(image)

    detected = doc_corners is not None
    debug_info['document_detected'] = detected

    if doc_corners is not None:
        print(f"[DEBUG] Document DETECTE - 4 coins trouves")
        print(f"[DEBUG] Coins: {doc_corners.tolist()}")
        # 2. Appliquer la transformation de perspective
        warped = four_point_transform(image, doc_corners)
        debug_info['after_transform'] = f"{warped.shape[1]}x{warped.shape[0]}"
        print(f"[DEBUG] Apres transform: {warped.shape[1]}x{warped.shape[0]}")
    else:
        print(f"[DEBUG] Document NON DETECTE - fallback trim_to_content")
        # Pas de document detecte - cropper quand meme au contenu
        warped = trim_to_content(image)
        debug_info['after_fallback_trim'] = f"{warped.shape[1]}x{warped.shape[0]}"
        print(f"[DEBUG] Apres fallback trim: {warped.shape[1]}x{warped.shape[0]}")

    # 3. Ameliorer l'image (optionnel)
    if enhance:
        result = enhance_document(warped)
        debug_info['after_enhance'] = f"{result.shape[1]}x{result.shape[0]}"
        print(f"[DEBUG] Apres enhance: {result.shape[1]}x{result.shape[0]}")
    else:
        # Meme sans enhance, cropper au contenu
        result = trim_to_content(warped)
        debug_info['after_final_trim'] = f"{result.shape[1]}x{result.shape[0]}"
        print(f"[DEBUG] Apres final trim: {result.shape[1]}x{result.shape[0]}")

    # 4. Encoder en base64 (JPEG haute qualite)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    _, buffer = cv2.imencode('.jpg', result, encode_params)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    print(f"[DEBUG] Final: {result.shape[1]}x{result.shape[0]}")
    print(f"[DEBUG] Reduction: {image.shape[1]}x{image.shape[0]} -> {result.shape[1]}x{result.shape[0]}")

    return {
        'processedImage': f"data:image/jpeg;base64,{result_base64}",
        'documentDetected': detected,
        'originalWidth': image.shape[1],
        'originalHeight': image.shape[0],
        'processedWidth': result.shape[1],
        'processedHeight': result.shape[0],
        'debug': debug_info,
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


@app.route('/debug', methods=['POST'])
def debug_scan():
    """Endpoint de debug - retourne les infos sans l'image"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Image manquante'}), 400

        image_base64 = data['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Image invalide'}), 400

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Analyse de l'image
        analysis = {
            'dimensions': f"{w}x{h}",
            'luminosity': {
                'min': int(gray.min()),
                'max': int(gray.max()),
                'mean': float(gray.mean()),
                'std': float(gray.std()),
            },
            'corners_sample': {
                'top_left_10x10': float(gray[0:10, 0:10].mean()),
                'top_right_10x10': float(gray[0:10, w-10:w].mean()),
                'bottom_left_10x10': float(gray[h-10:h, 0:10].mean()),
                'bottom_right_10x10': float(gray[h-10:h, w-10:w].mean()),
                'center_50x50': float(gray[h//2-25:h//2+25, w//2-25:w//2+25].mean()),
            }
        }

        # Test detection document
        doc_corners = detect_document(image)
        analysis['document_detected'] = doc_corners is not None
        if doc_corners is not None:
            analysis['document_corners'] = doc_corners.tolist()

        # Test seuils
        for thresh in [180, 200, 220, 235]:
            _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            pct = 100 * np.sum(mask > 0) / (h * w)
            analysis[f'pixels_above_{thresh}'] = f"{pct:.1f}%"

        return jsonify({
            'success': True,
            'analysis': analysis
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
